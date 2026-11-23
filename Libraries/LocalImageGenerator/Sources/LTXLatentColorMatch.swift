import Foundation
import NNC

/// Color stabilization of LTX video latents (prototype, fork only).
///
/// Both modes work per latent frame and channel, blended with `strength` (0 leaves the latents
/// unchanged, 1 applies the full correction).
/// - `first`: full AdaIN (mean and standard deviation) towards the encoded first frame, as
///   Lightricks' `LTXVAdainLatent` node (ComfyUI-LTXVideo, latent_norm.py) with `per_frame`
///   enabled. It removes slow drift, but it also flattens real lighting changes, and rescaling
///   the standard deviation alters texture as well as color.
/// - `smooth`: only the mean of each channel is moved, towards its moving average over `radius`
///   latent frames on each side (one latent frame is 8 video frames). Short jumps of brightness or
///   color cast, such as the beat at the 8-frame latent boundaries, are evened out, slow real
///   changes are kept, and the spread of the latents (which carries texture) is not touched: a
///   first version that also matched the standard deviation made the skin blotchy from one block
///   to the next. The first latent frame (the conditioning image) is left as it is.
/// Only the video rows (the first `videoHeight` rows of each frame) are touched; packed audio rows
/// below them are copied as they are.
enum LTXColorMatchMode: String {
  case first
  case smooth
}

struct LTXColorMatchSettings {
  var mode: LTXColorMatchMode
  var strength: Float
  var radius: Int

  /// Prototype plumbing: the CLI writes "mode:strength:radius" to DRAWTHINGS_LTX_COLOR_MATCH.
  static func fromEnvironment() -> LTXColorMatchSettings? {
    guard let value = ProcessInfo.processInfo.environment["DRAWTHINGS_LTX_COLOR_MATCH"] else {
      return nil
    }
    let parts = value.split(separator: ":").map(String.init)
    guard let first = parts.first, let mode = LTXColorMatchMode(rawValue: first) else {
      return nil
    }
    let strength = parts.count > 1 ? (Float(parts[1]) ?? 1) : 1
    let radius = parts.count > 2 ? (Int(parts[2]) ?? 2) : 2
    guard strength > 0 else { return nil }
    return LTXColorMatchSettings(mode: mode, strength: min(strength, 1), radius: max(radius, 1))
  }
}

func LTXLatentColorMatch<FloatType: TensorNumeric & BinaryFloatingPoint>(
  _ x: DynamicGraph.Tensor<FloatType>, reference: DynamicGraph.Tensor<FloatType>?,
  videoHeight: Int, settings: LTXColorMatchSettings
) -> DynamicGraph.Tensor<FloatType> {
  let shape = x.shape  // [frames, videoHeight (+ audio rows), width, channels]
  guard shape.count == 4, videoHeight > 0, videoHeight <= shape[1], shape[0] > 1 else {
    fputs("LTX color match skipped: latents \(shape), video height \(videoHeight)\n", stderr)
    return x
  }
  let frames = shape[0]
  let height = shape[1]
  let width = shape[2]
  let channels = shape[3]
  let plane = videoHeight * width
  let graph = x.graph
  let input = x.copied().rawValue.toCPU()

  // Mean and standard deviation of each channel over the video rows of a frame.
  func statistics(_ values: UnsafeBufferPointer<FloatType>, frameBase: Int) -> ([Double], [Double])
  {
    var sum = [Double](repeating: 0, count: channels)
    var sumOfSquares = [Double](repeating: 0, count: channels)
    for i in 0..<plane {
      let base = frameBase + i * channels
      for c in 0..<channels {
        let value = Double(values[base + c])
        sum[c] += value
        sumOfSquares[c] += value * value
      }
    }
    var mean = [Double](repeating: 0, count: channels)
    var std = [Double](repeating: 0, count: channels)
    for c in 0..<channels {
      mean[c] = sum[c] / Double(plane)
      std[c] = max(sumOfSquares[c] / Double(plane) - mean[c] * mean[c], 0).squareRoot()
    }
    return (mean, std)
  }

  let frameSize = height * width * channels
  var frameMeans = [[Double]]()
  var frameStds = [[Double]]()
  var targetMeans = [[Double]]()
  var targetStds = [[Double]]()
  input.withUnsafeBytes { raw in
    let values = raw.bindMemory(to: FloatType.self)
    for f in 0..<frames {
      let (mean, std) = statistics(values, frameBase: f * frameSize)
      frameMeans.append(mean)
      frameStds.append(std)
    }
  }
  var firstFrame = 0
  switch settings.mode {
  case .first:
    guard let reference, reference.shape.count == 4, reference.shape[1] >= videoHeight,
      reference.shape[2] == width, reference.shape[3] == channels
    else {
      fputs(
        "LTX color match skipped: latents \(shape), reference \(reference?.shape ?? [])\n", stderr)
      return x
    }
    let referenceOnCPU = reference.copied().rawValue.toCPU()
    let (mean, std) = referenceOnCPU.withUnsafeBytes { raw in
      statistics(raw.bindMemory(to: FloatType.self), frameBase: 0)
    }
    targetMeans = Array(repeating: mean, count: frames)
    targetStds = Array(repeating: std, count: frames)
  case .smooth:
    // The conditioning image (latent frame 0) stays as it is and is left out of the averages.
    firstFrame = 1
    targetMeans = frameMeans
    targetStds = frameStds
    for f in 1..<frames {
      let lower = max(1, f - settings.radius)
      let upper = min(frames - 1, f + settings.radius)
      let count = Double(upper - lower + 1)
      for c in 0..<channels {
        var mean = 0.0
        var std = 0.0
        for g in lower...upper {
          mean += frameMeans[g][c]
          std += frameStds[g][c]
        }
        targetMeans[f][c] = mean / count
        targetStds[f][c] = std / count
      }
    }
  }

  let count = frames * frameSize
  var output = [FloatType](repeating: 0, count: count)
  let blend = Double(settings.strength)
  input.withUnsafeBytes { raw in
    let values = raw.bindMemory(to: FloatType.self)
    for i in 0..<count {
      output[i] = values[i]
    }
    for f in firstFrame..<frames {
      let frameBase = f * frameSize
      var scale = [Double](repeating: 1, count: channels)
      if settings.mode == .first {
        for c in 0..<channels {
          scale[c] = targetStds[f][c] / max(frameStds[f][c], 1e-6)
        }
      }
      for i in 0..<plane {
        let base = frameBase + i * channels
        for c in 0..<channels {
          let value = Double(values[base + c])
          let matched = (value - frameMeans[f][c]) * scale[c] + targetMeans[f][c]
          output[base + c] = FloatType(value + blend * (matched - value))
        }
      }
    }
  }
  let result = Tensor<FloatType>(output, .CPU, .NHWC(frames, height, width, channels))
  return graph.variable(result.toGPU(0))
}

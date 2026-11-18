import Foundation
import NNC

/// Per-frame AdaIN of LTX video latents towards a reference frame latent, as Lightricks'
/// `LTXVAdainLatent` node (ComfyUI-LTXVideo, latent_norm.py) with `per_frame` enabled and a
/// one-frame reference (the encoded first frame): for every latent frame and channel, the mean
/// and standard deviation over the frame are matched to those of the reference, then blended
/// with `factor` (0 leaves the latents unchanged, 1 applies the full match). It counters the
/// slow drift of brightness and color cast that LTX shows, mostly at high resolution and in the
/// second pass. Only the video rows (the first `videoHeight` rows of each frame) are touched;
/// packed audio rows below them are copied as they are.
func LTXLatentColorMatch<FloatType: TensorNumeric & BinaryFloatingPoint>(
  _ x: DynamicGraph.Tensor<FloatType>, reference: DynamicGraph.Tensor<FloatType>,
  videoHeight: Int, factor: Float
) -> DynamicGraph.Tensor<FloatType> {
  guard factor != 0 else { return x }
  let shape = x.shape  // [frames, videoHeight (+ audio rows), width, channels]
  let referenceShape = reference.shape  // [1, videoHeight, width, channels]
  guard shape.count == 4, referenceShape.count == 4, videoHeight > 0, videoHeight <= shape[1],
    referenceShape[1] >= videoHeight, referenceShape[2] == shape[2],
    referenceShape[3] == shape[3]
  else {
    fputs(
      "LTX color match skipped: latents \(shape), reference \(referenceShape), video height \(videoHeight)\n",
      stderr)
    return x
  }
  let frames = shape[0]
  let height = shape[1]
  let width = shape[2]
  let channels = shape[3]
  let plane = videoHeight * width
  let graph = x.graph
  let input = x.copied().rawValue.toCPU()
  let referenceOnCPU = reference.copied().rawValue.toCPU()

  // Mean and standard deviation of each channel over the reference frame.
  var referenceMean = [Double](repeating: 0, count: channels)
  var referenceStd = [Double](repeating: 0, count: channels)
  referenceOnCPU.withUnsafeBytes { raw in
    let values = raw.bindMemory(to: FloatType.self)
    var sum = [Double](repeating: 0, count: channels)
    var sumOfSquares = [Double](repeating: 0, count: channels)
    for i in 0..<plane {
      let base = i * channels
      for c in 0..<channels {
        let value = Double(values[base + c])
        sum[c] += value
        sumOfSquares[c] += value * value
      }
    }
    for c in 0..<channels {
      let mean = sum[c] / Double(plane)
      referenceMean[c] = mean
      referenceStd[c] = max(sumOfSquares[c] / Double(plane) - mean * mean, 0).squareRoot()
    }
  }

  let count = frames * height * width * channels
  var output = [FloatType](repeating: 0, count: count)
  let blend = Double(factor)
  input.withUnsafeBytes { raw in
    let values = raw.bindMemory(to: FloatType.self)
    for i in 0..<count {
      output[i] = values[i]
    }
    var sum = [Double](repeating: 0, count: channels)
    var sumOfSquares = [Double](repeating: 0, count: channels)
    var mean = [Double](repeating: 0, count: channels)
    var scale = [Double](repeating: 0, count: channels)
    for f in 0..<frames {
      let frameBase = f * height * width * channels
      for c in 0..<channels {
        sum[c] = 0
        sumOfSquares[c] = 0
      }
      for i in 0..<plane {
        let base = frameBase + i * channels
        for c in 0..<channels {
          let value = Double(values[base + c])
          sum[c] += value
          sumOfSquares[c] += value * value
        }
      }
      for c in 0..<channels {
        mean[c] = sum[c] / Double(plane)
        let std = max(sumOfSquares[c] / Double(plane) - mean[c] * mean[c], 0).squareRoot()
        scale[c] = referenceStd[c] / max(std, 1e-6)
      }
      for i in 0..<plane {
        let base = frameBase + i * channels
        for c in 0..<channels {
          let value = Double(values[base + c])
          let matched = (value - mean[c]) * scale[c] + referenceMean[c]
          output[base + c] = FloatType(value + blend * (matched - value))
        }
      }
    }
  }
  let result = Tensor<FloatType>(output, .CPU, .NHWC(frames, height, width, channels))
  return graph.variable(result.toGPU(0))
}

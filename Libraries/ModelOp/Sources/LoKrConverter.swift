import Diffusion
import Foundation
import NNC

#if canImport(Accelerate)
  import Accelerate
#endif

/// Rewrites LyCORIS LoKr adapters (`lokr_w1`, `lokr_w2` and their low-rank `_a` / `_b` forms) as
/// low-rank LoRA pairs, so that the rest of the importer reads them like any PEFT LoRA.
///
/// A LoKr module adds `scale * kron(W1, W2)` to its weight. Draw Things stores LoRAs as up / down
/// pairs, so each update is written at a chosen rank from its singular value decomposition. That
/// decomposition comes exactly from the decompositions of the two small factors: the singular
/// values of `kron(W1, W2)` are the products of theirs, with Kronecker products of their singular
/// vectors. Keeping the largest products is the best approximation at that rank (Eckart-Young), and
/// it is exact when the rank covers every nonzero product.
public enum LoKrConverter {
  public enum Error: Swift.Error, CustomStringConvertible {
    case tuckerNotSupported(String)
    case missingFactor(String)
    case shapeMismatch(String)
    case svdFailed(String, Int32)
    case unsupportedPlatform

    public var description: String {
      switch self {
      case .tuckerNotSupported(let module):
        return "LoKr module \(module) uses a Tucker core (lokr_t2), which is not supported yet."
      case .missingFactor(let module):
        return "LoKr module \(module) has an incomplete factor (lokr_w1 or lokr_w2)."
      case .shapeMismatch(let module):
        return "LoKr module \(module) has factors whose shapes do not match."
      case .svdFailed(let module, let info):
        return "The singular value decomposition failed for LoKr module \(module) (info \(info))."
      case .unsupportedPlatform:
        return "LoKr import needs Accelerate (macOS or iOS)."
      }
    }
  }

  public static let defaultRank = 128

  /// A summary of the last conversion (modules, rank, share of the update kept), or nil when the
  /// last import had no LoKr module. Read by the CLI after an import.
  public static var lastReport: String? = nil

  private static let suffixes = [
    "lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b", "lokr_t2",
  ]

  /// Replaces every LoKr module of `stateDict` by `<module>.lora_A.weight` (down) and
  /// `<module>.lora_B.weight` (up), with the scale folded into the up projection and no alpha.
  /// Returns the archive to read the state dictionary from afterwards: `archive` itself when there
  /// is no LoKr module.
  public static func convert(
    stateDict: inout [String: TensorDescriptor], archive: TensorArchive, rank: Int
  ) throws -> TensorArchive {
    lastReport = nil
    var modules = Set<String>()
    for key in stateDict.keys {
      for suffix in suffixes where key.hasSuffix("." + suffix) {
        modules.insert(String(key.dropLast(suffix.count + 1)))
      }
    }
    guard !modules.isEmpty else { return archive }
    #if canImport(Accelerate)
      let overlay = OverlayTensorArchive(base: archive)
      let rank = max(1, rank)
      var kept = [Double]()
      var usedRanks = [Int]()
      var skippedDoRA = 0
      for module in modules.sorted() {
        func read(_ suffix: String) throws -> (values: [Float], shape: [Int])? {
          guard let descriptor = stateDict[module + "." + suffix] else { return nil }
          return try archive.with(descriptor) {
            let tensor = Tensor<Float>(from: $0).copied()
            let shape = Array(tensor.shape)
            let count = shape.reduce(1, *)
            let values = tensor.withUnsafeBytes {
              Array(UnsafeBufferPointer(start: $0.bindMemory(to: Float.self).baseAddress, count: count))
            }
            return (values, shape)
          }
        }
        guard stateDict[module + ".lokr_t2"] == nil else {
          throw Error.tuckerNotSupported(module)
        }
        // As in LyCORIS and ComfyUI: alpha / rank applies only when a factor is low-rank (the rank
        // of the last low-rank factor), otherwise the update is used as is.
        var dim: Int? = nil
        let w1: [Float]
        let w1Shape: [Int]
        if let full = try read("lokr_w1") {
          w1 = full.values
          w1Shape = full.shape
        } else if let a = try read("lokr_w1_a"), let b = try read("lokr_w1_b") {
          guard a.shape.count == 2, b.shape.count == 2, a.shape[1] == b.shape[0] else {
            throw Error.shapeMismatch(module)
          }
          w1 = multiply(a.values, b.values, m: a.shape[0], k: a.shape[1], n: b.shape[1])
          w1Shape = [a.shape[0], b.shape[1]]
          dim = b.shape[0]
        } else {
          throw Error.missingFactor(module)
        }
        // A convolution factor (out, in, kh, kw) is flattened to (out, in * kh * kw): the Kronecker
        // product with the 2-D first factor then lines up with the flattened weight.
        var w2: [Float]
        var w2Shape: [Int]
        if let full = try read("lokr_w2") {
          w2 = full.values
          w2Shape = full.shape
        } else if let a = try read("lokr_w2_a"), let b = try read("lokr_w2_b") {
          let columns = b.shape.dropFirst().reduce(1, *)
          guard a.shape.count == 2, a.shape[1] == b.shape[0] else {
            throw Error.shapeMismatch(module)
          }
          w2 = multiply(a.values, b.values, m: a.shape[0], k: a.shape[1], n: columns)
          w2Shape = [a.shape[0]] + Array(b.shape.dropFirst())
          dim = b.shape[0]
        } else {
          throw Error.missingFactor(module)
        }
        guard w1Shape.count == 2, w2Shape.count == 2 || w2Shape.count == 4 else {
          throw Error.shapeMismatch(module)
        }
        var scale: Float = 1
        if let dim = dim, let alpha = try read("alpha")?.values.first {
          scale = alpha / Float(dim)
        }
        if stateDict[module + ".dora_scale"] != nil {
          skippedDoRA += 1
        }
        let (o1, i1) = (w1Shape[0], w1Shape[1])
        let o2 = w2Shape[0]
        let n2 = w2Shape.dropFirst().reduce(1, *)
        let kernel = Array(w2Shape.dropFirst(2))
        let a = try svd(w1, rows: o1, columns: i1, module: module)
        let b = try svd(w2, rows: o2, columns: n2, module: module)
        let k1 = a.s.count
        let k2 = b.s.count
        var products = [(value: Float, i: Int, j: Int)]()
        products.reserveCapacity(k1 * k2)
        var total: Double = 0
        for i in 0..<k1 {
          for j in 0..<k2 {
            let value = a.s[i] * b.s[j]
            total += Double(value) * Double(value)
            if value > 0 {
              products.append((value, i, j))
            }
          }
        }
        products.sort { $0.value > $1.value }
        let r = max(1, min(rank, products.count))
        var retained: Double = 0
        for t in 0..<min(r, products.count) {
          retained += Double(products[t].value) * Double(products[t].value)
        }
        kept.append(total > 0 ? retained / total : 1)
        usedRanks.append(r)
        let outFeatures = o1 * o2
        let inFeatures = i1 * n2
        var up = Tensor<Float>(
          .CPU, format: .NCHW, shape: TensorShape(kernel.isEmpty ? [outFeatures, r] : [outFeatures, r, 1, 1]))
        var down = Tensor<Float>(
          .CPU, format: .NCHW,
          shape: TensorShape(kernel.isEmpty ? [r, inFeatures] : [r, i1 * w2Shape[1]] + kernel))
        up.withUnsafeMutableBytes {
          let up = $0.bindMemory(to: Float.self)
          for index in 0..<up.count { up[index] = 0 }
          for t in 0..<min(r, products.count) {
            let (value, i, j) = products[t]
            let factor = value.squareRoot() * scale
            for p in 0..<o1 {
              let left = a.u[p * k1 + i] * factor
              guard left != 0 else { continue }
              for q in 0..<o2 {
                up[(p * o2 + q) * r + t] = left * b.u[q * k2 + j]
              }
            }
          }
        }
        down.withUnsafeMutableBytes {
          let down = $0.bindMemory(to: Float.self)
          for index in 0..<down.count { down[index] = 0 }
          for t in 0..<min(r, products.count) {
            let (value, i, j) = products[t]
            let factor = value.squareRoot()
            for c in 0..<i1 {
              let left = a.vt[i * i1 + c] * factor
              guard left != 0 else { continue }
              for d in 0..<n2 {
                down[t * inFeatures + c * n2 + d] = left * b.vt[j * n2 + d]
              }
            }
          }
        }
        for suffix in suffixes + ["alpha", "dora_scale"] {
          stateDict[module + "." + suffix] = nil
        }
        stateDict[module + ".lora_A.weight"] = overlay.add(down)
        stateDict[module + ".lora_B.weight"] = overlay.add(up)
      }
      let mean = kept.reduce(0, +) / Double(kept.count)
      let worst = kept.min() ?? 1
      var report =
        "LoKr: \(modules.count) modules converted to LoRA at rank \(usedRanks.max() ?? rank)"
        + String(
          format: " (update kept: %.1f%% on average, %.1f%% at worst)", mean * 100, worst * 100)
      if skippedDoRA > 0 {
        report += "; DoRA scales ignored on \(skippedDoRA) modules"
      }
      lastReport = report
      return overlay
    #else
      throw Error.unsupportedPlatform
    #endif
  }

  #if canImport(Accelerate)
    /// Row-major product of an m x k matrix by a k x n matrix.
    private static func multiply(_ a: [Float], _ b: [Float], m: Int, k: Int, n: Int) -> [Float] {
      var c = [Float](repeating: 0, count: m * n)
      cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(m), Int32(n), Int32(k), 1, a, Int32(k),
        b, Int32(n), 0, &c, Int32(n))
      return c
    }

    /// Thin SVD of a row-major matrix: u is rows x k and vt is k x columns, both row-major, with
    /// k = min(rows, columns) and s in decreasing order.
    ///
    /// LAPACK is column-major: the row-major matrix is the column-major transpose, so this
    /// decomposes the transpose and reads its factors back swapped.
    private static func svd(_ matrix: [Float], rows: Int, columns: Int, module: String) throws
      -> (u: [Float], s: [Float], vt: [Float])
    {
      let k = min(rows, columns)
      var a = matrix
      var jobz = CChar(UInt8(ascii: "S"))
      var m = __CLPK_integer(columns)
      var n = __CLPK_integer(rows)
      var lda = m
      var s = [Float](repeating: 0, count: k)
      // Factors of the transpose: its left vectors (columns x k) are our right vectors.
      var transposedU = [Float](repeating: 0, count: columns * k)
      var transposedVt = [Float](repeating: 0, count: k * rows)
      var ldu = m
      var ldvt = __CLPK_integer(k)
      var iwork = [__CLPK_integer](repeating: 0, count: 8 * k)
      var info: __CLPK_integer = 0
      var lwork: __CLPK_integer = -1
      var optimal: Float = 0
      sgesdd_(
        &jobz, &m, &n, &a, &lda, &s, &transposedU, &ldu, &transposedVt, &ldvt, &optimal, &lwork,
        &iwork, &info)
      guard info == 0 else { throw Error.svdFailed(module, info) }
      lwork = __CLPK_integer(optimal)
      var work = [Float](repeating: 0, count: Int(lwork))
      sgesdd_(
        &jobz, &m, &n, &a, &lda, &s, &transposedU, &ldu, &transposedVt, &ldvt, &work, &lwork,
        &iwork, &info)
      guard info == 0 else { throw Error.svdFailed(module, info) }
      // transposedVt is k x rows column-major, that is rows x k row-major: our u.
      // transposedU is columns x k column-major, that is k x columns row-major: our vt.
      return (transposedVt, s, transposedU)
    }
  #endif
}

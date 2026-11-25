import NNC

/// A tensor archive that serves tensors computed in memory next to the tensors of another archive.
/// Importers use it to rewrite a weight format into one they already read (for example a LoKr
/// adapter into low-rank LoRA up / down pairs) without changing how the rest of the import works.
public final class OverlayTensorArchive: TensorArchive {
  private static let prefix = "__overlay__"
  private let base: TensorArchive
  private var tensors = [String: Tensor<Float>]()

  public init(base: TensorArchive) {
    self.base = base
  }

  /// Keeps a copy of the tensor and returns a descriptor that this archive resolves to it. The
  /// tensor must be contiguous (as created by `Tensor<Float>(.CPU, format:shape:)`).
  public func add(_ tensor: Tensor<Float>) -> TensorDescriptor {
    let shape = Array(tensor.shape)
    let name = "\(Self.prefix)\(tensors.count)"
    tensors[name] = tensor
    var strides = [Int]()
    var stride = 1
    for dimension in shape.reversed() {
      strides.append(stride)
      stride *= dimension
    }
    strides.reverse()
    return TensorDescriptor(
      storage: Storage(
        name: name, size: stride * MemoryLayout<Float>.size, dataType: .Float32, BF16: false,
        FP8_E4M3: false, FP8_E5M2: false),
      storageOffset: 0, shape: shape, strides: strides)
  }

  public func with<T>(_ tensorDescriptor: TensorDescriptor, block: (AnyTensor) throws -> T) throws
    -> T
  {
    if let tensor = tensors[tensorDescriptor.storage.name] {
      return try block(tensor)
    }
    return try base.with(tensorDescriptor, block: block)
  }
}

/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace pto {
class PtrType;
class TensorViewType;
class PartitionTensorViewType;
class TileType;
class TileBufType;
namespace detail {
struct PtrTypeStorage;
} // namespace detail
class PtrType : public ::mlir::Type::TypeBase<PtrType, ::mlir::Type, detail::PtrTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "pto.ptr";
  static constexpr ::llvm::StringLiteral dialectName = "pto";
  static PtrType get(::mlir::MLIRContext *context, Type elementType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ptr"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  mlir::Type getElementType() const;
};
namespace detail {
struct TensorViewTypeStorage;
} // namespace detail
class TensorViewType : public ::mlir::Type::TypeBase<TensorViewType, ::mlir::Type, detail::TensorViewTypeStorage> {
public:
  using Base::Base;
  int64_t getRank() const { return getShape().size(); }
  int64_t getDimSize(unsigned idx) const { return getShape()[idx]; }
  int64_t getNumElements() const {
    int64_t num = 1;
    for (int64_t dim : getShape()) {
      if (dim == mlir::ShapedType::kDynamic) return mlir::ShapedType::kDynamic;
      num *= dim;
    }
    return num;
  }
  static constexpr ::llvm::StringLiteral name = "pto.tensor_view";
  static constexpr ::llvm::StringLiteral dialectName = "pto";
  static TensorViewType get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType);
  static TensorViewType get(::mlir::MLIRContext *context, int64_t rank, Type elementType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensor_view"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
  mlir::Type getElementType() const;
};
namespace detail {
struct PartitionTensorViewTypeStorage;
} // namespace detail
class PartitionTensorViewType : public ::mlir::Type::TypeBase<PartitionTensorViewType, ::mlir::Type, detail::PartitionTensorViewTypeStorage> {
public:
  using Base::Base;
  int64_t getRank() const { return getShape().size(); }
  int64_t getDimSize(unsigned idx) const { return getShape()[idx]; }

  int64_t getNumElements() const {
    int64_t num = 1;
    for (int64_t dim : getShape()) {
      if (dim == mlir::ShapedType::kDynamic) return mlir::ShapedType::kDynamic;
      num *= dim;
    }
    return num;
  }
  static constexpr ::llvm::StringLiteral name = "pto.partition_tensor_view";
  static constexpr ::llvm::StringLiteral dialectName = "pto";
  static PartitionTensorViewType get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"partition_tensor_view"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
  mlir::Type getElementType() const;
};
namespace detail {
struct TileTypeStorage;
} // namespace detail
class TileType : public ::mlir::Type::TypeBase<TileType, ::mlir::Type, detail::TileTypeStorage> {
public:
  using Base::Base;
  int64_t getRank() const { return getShape().size(); }
  int64_t getDimSize(unsigned idx) const { return getShape()[idx]; }
  int64_t getNumElements() const {
    int64_t num = 1;
    for (int64_t dim : getShape()) num *= dim;
    return num;
  }
  mlir::pto::TileBufType toBuffer(
      mlir::Attribute memorySpace,
      ArrayRef<int64_t> validShape = {},
      mlir::pto::TileBufConfigAttr config = {}) const;
  static constexpr ::llvm::StringLiteral name = "pto.tile";
  static constexpr ::llvm::StringLiteral dialectName = "pto";
  static TileType get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tile"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
  mlir::Type getElementType() const;
};
namespace detail {
struct TileBufTypeStorage;
} // namespace detail
class TileBufType : public ::mlir::Type::TypeBase<TileBufType, ::mlir::Type, detail::TileBufTypeStorage> {
public:
  using Base::Base;
  int64_t getRank() const { return getShape().size(); }

  bool hasValidShape() const { return !getValidShape().empty(); }
  bool hasDynamicValid() const {
    auto vs = getValidShape();
    return vs.size()==2 && vs[0] < 0 && vs[1] < 0;
  }

  mlir::pto::TileBufConfigAttr getConfigAttr() const;
  bool hasNonDefaultConfig() const;

  // ✅ 返回强类型 Attr，而不是 enum
  mlir::Attribute getBLayoutAttr() const;
  mlir::Attribute getSLayoutAttr() const;
  int32_t getSFractalSizeI32() const;
  mlir::Attribute getPadValueAttr() const;

  // 如果你仍然想要“数值枚举”，就提供 int getter（不会依赖 enum 类型）
  int32_t getBLayoutValueI32() const;   // 0 row_major, 1 col_major
  int32_t getSLayoutValueI32() const;   // 0 none_box, 1 row_major, 2 col_major
  int32_t getPadValueI32() const;       // 0 null, 1 zero, 2 max, 3 min
  static constexpr ::llvm::StringLiteral name = "pto.tile_buf";
  static constexpr ::llvm::StringLiteral dialectName = "pto";
  static TileBufType get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType, mlir::Attribute memorySpace, ::llvm::ArrayRef<int64_t> validShape, mlir::pto::TileBufConfigAttr config);
  static TileBufType get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape, Type elementType, Attribute memorySpace);
  static TileBufType get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape, Type elementType, Attribute memorySpace, mlir::pto::TileBufConfigAttr config);
  static TileBufType get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape, Type elementType, Attribute memorySpace, ArrayRef<int64_t> validShape);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tile_buf"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
  mlir::Type getElementType() const;
  mlir::Attribute getMemorySpace() const;
  ::llvm::ArrayRef<int64_t> getValidShape() const;
  mlir::pto::TileBufConfigAttr getConfig() const;
};
} // namespace pto
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::pto::PtrType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::pto::TensorViewType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::pto::PartitionTensorViewType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::pto::TileType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::pto::TileBufType)

#endif  // GET_TYPEDEF_CLASSES


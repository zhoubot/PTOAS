/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::pto::PtrType,
::mlir::pto::TensorViewType,
::mlir::pto::PartitionTensorViewType,
::mlir::pto::TileType,
::mlir::pto::TileBufType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::pto::PtrType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::PtrType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::TensorViewType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::TensorViewType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::PartitionTensorViewType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::PartitionTensorViewType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::TileType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::TileType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::TileBufType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::TileBufType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::pto::PtrType>([&](auto t) {
      printer << ::mlir::pto::PtrType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::TensorViewType>([&](auto t) {
      printer << ::mlir::pto::TensorViewType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::PartitionTensorViewType>([&](auto t) {
      printer << ::mlir::pto::PartitionTensorViewType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::TileType>([&](auto t) {
      printer << ::mlir::pto::TileType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::TileBufType>([&](auto t) {
      printer << ::mlir::pto::TileBufType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace pto {
namespace detail {
struct PtrTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<mlir::Type>;
  PtrTypeStorage(mlir::Type elementType) : elementType(std::move(elementType)) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static PtrTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto elementType = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<PtrTypeStorage>()) PtrTypeStorage(std::move(elementType));
  }

  mlir::Type elementType;
};
} // namespace detail
PtrType PtrType::get(::mlir::MLIRContext *context, Type elementType) {
  return Base::get(context, elementType);
}

::mlir::Type PtrType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<mlir::Type> _result_elementType;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'elementType'
  _result_elementType = ::mlir::FieldParser<mlir::Type>::parse(odsParser);
  if (::mlir::failed(_result_elementType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PtrType parameter 'elementType' which is to be a `mlir::Type`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_elementType));
  return PtrType::get(odsParser.getContext(),
      mlir::Type((*_result_elementType)));
}

void PtrType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getElementType());
  odsPrinter << ">";
}

mlir::Type PtrType::getElementType() const {
  return getImpl()->elementType;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::PtrType)
namespace mlir {
namespace pto {
namespace detail {
struct TensorViewTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, mlir::Type>;
  TensorViewTypeStorage(::llvm::ArrayRef<int64_t> shape, mlir::Type elementType) : shape(std::move(shape)), elementType(std::move(elementType)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, elementType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static TensorViewTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto elementType = std::move(std::get<1>(tblgenKey));
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<TensorViewTypeStorage>()) TensorViewTypeStorage(std::move(shape), std::move(elementType));
  }

  ::llvm::ArrayRef<int64_t> shape;
  mlir::Type elementType;
};
} // namespace detail
TensorViewType TensorViewType::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType) {
  return Base::get(context, std::move(shape), std::move(elementType));
}

TensorViewType TensorViewType::get(::mlir::MLIRContext *context, int64_t rank, Type elementType) {
  SmallVector<int64_t, 4> shp(rank, mlir::ShapedType::kDynamic);
  return Base::get(context, shp, elementType);
}

::llvm::ArrayRef<int64_t> TensorViewType::getShape() const {
  return getImpl()->shape;
}

mlir::Type TensorViewType::getElementType() const {
  return getImpl()->elementType;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::TensorViewType)
namespace mlir {
namespace pto {
namespace detail {
struct PartitionTensorViewTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, mlir::Type>;
  PartitionTensorViewTypeStorage(::llvm::ArrayRef<int64_t> shape, mlir::Type elementType) : shape(std::move(shape)), elementType(std::move(elementType)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, elementType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static PartitionTensorViewTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto elementType = std::move(std::get<1>(tblgenKey));
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<PartitionTensorViewTypeStorage>()) PartitionTensorViewTypeStorage(std::move(shape), std::move(elementType));
  }

  ::llvm::ArrayRef<int64_t> shape;
  mlir::Type elementType;
};
} // namespace detail
PartitionTensorViewType PartitionTensorViewType::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType) {
  return Base::get(context, std::move(shape), std::move(elementType));
}

::llvm::ArrayRef<int64_t> PartitionTensorViewType::getShape() const {
  return getImpl()->shape;
}

mlir::Type PartitionTensorViewType::getElementType() const {
  return getImpl()->elementType;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::PartitionTensorViewType)
namespace mlir {
namespace pto {
namespace detail {
struct TileTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, mlir::Type>;
  TileTypeStorage(::llvm::ArrayRef<int64_t> shape, mlir::Type elementType) : shape(std::move(shape)), elementType(std::move(elementType)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, elementType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static TileTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto elementType = std::move(std::get<1>(tblgenKey));
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<TileTypeStorage>()) TileTypeStorage(std::move(shape), std::move(elementType));
  }

  ::llvm::ArrayRef<int64_t> shape;
  mlir::Type elementType;
};
} // namespace detail
TileType TileType::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType) {
  return Base::get(context, std::move(shape), std::move(elementType));
}

::llvm::ArrayRef<int64_t> TileType::getShape() const {
  return getImpl()->shape;
}

mlir::Type TileType::getElementType() const {
  return getImpl()->elementType;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::TileType)
namespace mlir {
namespace pto {
namespace detail {
struct TileBufTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, mlir::Type, mlir::Attribute, ::llvm::ArrayRef<int64_t>, mlir::pto::TileBufConfigAttr>;
  TileBufTypeStorage(::llvm::ArrayRef<int64_t> shape, mlir::Type elementType, mlir::Attribute memorySpace, ::llvm::ArrayRef<int64_t> validShape, mlir::pto::TileBufConfigAttr config) : shape(std::move(shape)), elementType(std::move(elementType)), memorySpace(std::move(memorySpace)), validShape(std::move(validShape)), config(std::move(config)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, elementType, memorySpace, validShape, config);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey)) && (memorySpace == std::get<2>(tblgenKey)) && (validShape == std::get<3>(tblgenKey)) && (config == std::get<4>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey));
  }

  static TileBufTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto elementType = std::move(std::get<1>(tblgenKey));
    auto memorySpace = std::move(std::get<2>(tblgenKey));
    auto validShape = std::move(std::get<3>(tblgenKey));
    auto config = std::move(std::get<4>(tblgenKey));
    shape = allocator.copyInto(shape);
    validShape = allocator.copyInto(validShape);
    return new (allocator.allocate<TileBufTypeStorage>()) TileBufTypeStorage(std::move(shape), std::move(elementType), std::move(memorySpace), std::move(validShape), std::move(config));
  }

  ::llvm::ArrayRef<int64_t> shape;
  mlir::Type elementType;
  mlir::Attribute memorySpace;
  ::llvm::ArrayRef<int64_t> validShape;
  mlir::pto::TileBufConfigAttr config;
};
} // namespace detail
TileBufType TileBufType::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType, mlir::Attribute memorySpace, ::llvm::ArrayRef<int64_t> validShape, mlir::pto::TileBufConfigAttr config) {
  return Base::get(context, std::move(shape), std::move(elementType), std::move(memorySpace), std::move(validShape), std::move(config));
}

TileBufType TileBufType::get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape, Type elementType, Attribute memorySpace) {
  auto cfg = mlir::pto::TileBufConfigAttr::getDefault(context);

  // ✅ 默认 validShape = shape（你要求的语义）
  SmallVector<int64_t, 2> vs(shape.begin(), shape.end());

  return Base::get(context, shape, elementType, memorySpace, vs, cfg);
}

TileBufType TileBufType::get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape, Type elementType, Attribute memorySpace, mlir::pto::TileBufConfigAttr config) {
  // ✅ 默认 validShape = shape
  SmallVector<int64_t, 2> vs(shape.begin(), shape.end());
  return Base::get(context, shape, elementType, memorySpace, vs, config);
}

TileBufType TileBufType::get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape, Type elementType, Attribute memorySpace, ArrayRef<int64_t> validShape) {
  auto cfg = mlir::pto::TileBufConfigAttr::getDefault(context);
  return Base::get(context, shape, elementType, memorySpace, validShape, cfg);
}

::llvm::ArrayRef<int64_t> TileBufType::getShape() const {
  return getImpl()->shape;
}

mlir::Type TileBufType::getElementType() const {
  return getImpl()->elementType;
}

mlir::Attribute TileBufType::getMemorySpace() const {
  return getImpl()->memorySpace;
}

::llvm::ArrayRef<int64_t> TileBufType::getValidShape() const {
  return getImpl()->validShape;
}

mlir::pto::TileBufConfigAttr TileBufType::getConfig() const {
  return getImpl()->config;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::TileBufType)
namespace mlir {
namespace pto {

/// Parse a type registered to this dialect.
::mlir::Type PTODialect::parseType(::mlir::DialectAsmParser &parser) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  
  parser.emitError(typeLoc) << "unknown  type `"
      << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print a type registered to this dialect.
void PTODialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}
} // namespace pto
} // namespace mlir

#endif  // GET_TYPEDEF_CLASSES


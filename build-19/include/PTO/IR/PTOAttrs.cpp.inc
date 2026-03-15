/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::pto::AddressSpaceAttr,
::mlir::pto::PipeAttr,
::mlir::pto::PipeEventTypeAttr,
::mlir::pto::SyncOpTypeAttr,
::mlir::pto::LayoutAttr,
::mlir::pto::TFuncCoreTypeAttr,
::mlir::pto::TModuleCoreTypeAttr,
::mlir::pto::EventAttr,
::mlir::pto::PadModeAttr,
::mlir::pto::VectorFunctionAttr,
::mlir::pto::CmpModeAttr,
::mlir::pto::RoundModeAttr,
::mlir::pto::MaskPatternAttr,
::mlir::pto::BLayoutAttr,
::mlir::pto::SLayoutAttr,
::mlir::pto::PadValueAttr,
::mlir::pto::TileBufConfigAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::pto::AddressSpaceAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::AddressSpaceAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::PipeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::PipeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::PipeEventTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::PipeEventTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::SyncOpTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::SyncOpTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::LayoutAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::LayoutAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::TFuncCoreTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::TFuncCoreTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::TModuleCoreTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::TModuleCoreTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::EventAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::EventAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::PadModeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::PadModeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::VectorFunctionAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::VectorFunctionAttr::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::CmpModeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::CmpModeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::RoundModeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::RoundModeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::MaskPatternAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::MaskPatternAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::BLayoutAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::BLayoutAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::SLayoutAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::SLayoutAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::PadValueAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::PadValueAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pto::TileBufConfigAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pto::TileBufConfigAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::pto::AddressSpaceAttr>([&](auto t) {
      printer << ::mlir::pto::AddressSpaceAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::PipeAttr>([&](auto t) {
      printer << ::mlir::pto::PipeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::PipeEventTypeAttr>([&](auto t) {
      printer << ::mlir::pto::PipeEventTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::SyncOpTypeAttr>([&](auto t) {
      printer << ::mlir::pto::SyncOpTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::LayoutAttr>([&](auto t) {
      printer << ::mlir::pto::LayoutAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::TFuncCoreTypeAttr>([&](auto t) {
      printer << ::mlir::pto::TFuncCoreTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::TModuleCoreTypeAttr>([&](auto t) {
      printer << ::mlir::pto::TModuleCoreTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::EventAttr>([&](auto t) {
      printer << ::mlir::pto::EventAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::PadModeAttr>([&](auto t) {
      printer << ::mlir::pto::PadModeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::VectorFunctionAttr>([&](auto t) {
      printer << ::mlir::pto::VectorFunctionAttr::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::pto::CmpModeAttr>([&](auto t) {
      printer << ::mlir::pto::CmpModeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::RoundModeAttr>([&](auto t) {
      printer << ::mlir::pto::RoundModeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::MaskPatternAttr>([&](auto t) {
      printer << ::mlir::pto::MaskPatternAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::BLayoutAttr>([&](auto t) {
      printer << ::mlir::pto::BLayoutAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::SLayoutAttr>([&](auto t) {
      printer << ::mlir::pto::SLayoutAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::PadValueAttr>([&](auto t) {
      printer << ::mlir::pto::PadValueAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pto::TileBufConfigAttr>([&](auto t) {
      printer << ::mlir::pto::TileBufConfigAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace pto {
namespace detail {
struct AddressSpaceAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::pto::AddressSpace>;
  AddressSpaceAttrStorage(::mlir::pto::AddressSpace address_space) : address_space(std::move(address_space)) {}

  KeyTy getAsKey() const {
    return KeyTy(address_space);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (address_space == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static AddressSpaceAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto address_space = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<AddressSpaceAttrStorage>()) AddressSpaceAttrStorage(std::move(address_space));
  }

  ::mlir::pto::AddressSpace address_space;
};
} // namespace detail
AddressSpaceAttr AddressSpaceAttr::get(::mlir::MLIRContext *context, ::mlir::pto::AddressSpace address_space) {
  return Base::get(context, std::move(address_space));
}

::mlir::Attribute AddressSpaceAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::pto::AddressSpace> _result_address_space;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'address_space'
  _result_address_space = [&]() -> ::mlir::FailureOr<::mlir::pto::AddressSpace> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::pto::symbolizeAddressSpace(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::pto::AddressSpace" << " to be one of: " << "zero" << ", " << "gm" << ", " << "mat" << ", " << "left" << ", " << "right" << ", " << "acc" << ", " << "vec" << ", " << "bias" << ", " << "scaling")};
    }();
  if (::mlir::failed(_result_address_space)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PTO_AddressSpaceAttr parameter 'address_space' which is to be a `::mlir::pto::AddressSpace`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_address_space));
  return AddressSpaceAttr::get(odsParser.getContext(),
      ::mlir::pto::AddressSpace((*_result_address_space)));
}

void AddressSpaceAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyAddressSpace(getAddressSpace());
  }
  odsPrinter << ">";
}

::mlir::pto::AddressSpace AddressSpaceAttr::getAddressSpace() const {
  return getImpl()->address_space;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::AddressSpaceAttr)
namespace mlir {
namespace pto {
namespace detail {
struct PipeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::pto::PIPE>;
  PipeAttrStorage(::mlir::pto::PIPE pipe) : pipe(std::move(pipe)) {}

  KeyTy getAsKey() const {
    return KeyTy(pipe);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (pipe == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static PipeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto pipe = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<PipeAttrStorage>()) PipeAttrStorage(std::move(pipe));
  }

  ::mlir::pto::PIPE pipe;
};
} // namespace detail
PipeAttr PipeAttr::get(::mlir::MLIRContext *context, ::mlir::pto::PIPE pipe) {
  return Base::get(context, std::move(pipe));
}

::mlir::Attribute PipeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::pto::PIPE> _result_pipe;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'pipe'
  _result_pipe = [&]() -> ::mlir::FailureOr<::mlir::pto::PIPE> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::pto::symbolizePIPE(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::pto::PIPE" << " to be one of: " << "PIPE_S" << ", " << "PIPE_V" << ", " << "PIPE_M" << ", " << "PIPE_MTE1" << ", " << "PIPE_MTE2" << ", " << "PIPE_MTE3" << ", " << "PIPE_ALL" << ", " << "PIPE_MTE4" << ", " << "PIPE_MTE5" << ", " << "PIPE_V2" << ", " << "PIPE_FIX" << ", " << "VIRTUAL_PIPE_MTE2_L1A" << ", " << "VIRTUAL_PIPE_MTE2_L1B" << ", " << "PIPE_NUM" << ", " << "PIPE_UNASSIGNED")};
    }();
  if (::mlir::failed(_result_pipe)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PTO_PipeAttr parameter 'pipe' which is to be a `::mlir::pto::PIPE`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_pipe));
  return PipeAttr::get(odsParser.getContext(),
      ::mlir::pto::PIPE((*_result_pipe)));
}

void PipeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyPIPE(getPipe());
  }
  odsPrinter << ">";
}

::mlir::pto::PIPE PipeAttr::getPipe() const {
  return getImpl()->pipe;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::PipeAttr)
namespace mlir {
namespace pto {
namespace detail {
struct PipeEventTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::pto::SyncOpType>;
  PipeEventTypeAttrStorage(::mlir::pto::SyncOpType op_type) : op_type(std::move(op_type)) {}

  KeyTy getAsKey() const {
    return KeyTy(op_type);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (op_type == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static PipeEventTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto op_type = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<PipeEventTypeAttrStorage>()) PipeEventTypeAttrStorage(std::move(op_type));
  }

  ::mlir::pto::SyncOpType op_type;
};
} // namespace detail
PipeEventTypeAttr PipeEventTypeAttr::get(::mlir::MLIRContext *context, ::mlir::pto::SyncOpType op_type) {
  return Base::get(context, std::move(op_type));
}

::mlir::Attribute PipeEventTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::pto::SyncOpType> _result_op_type;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'op_type'
  _result_op_type = [&]() -> ::mlir::FailureOr<::mlir::pto::SyncOpType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::pto::symbolizeSyncOpType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::pto::SyncOpType" << " to be one of: " << "TLOAD" << ", " << "TSTORE_ACC" << ", " << "TSTORE_VEC" << ", " << "TMOV_M2L" << ", " << "TMOV_M2S" << ", " << "TMOV_M2B" << ", " << "TMOV_M2V" << ", " << "TMOV_V2M" << ", " << "TMATMUL" << ", " << "TVEC" << ", " << "TVECWAIT_EVENT")};
    }();
  if (::mlir::failed(_result_op_type)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PTO_PipeEventTypeAttr parameter 'op_type' which is to be a `::mlir::pto::SyncOpType`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_op_type));
  return PipeEventTypeAttr::get(odsParser.getContext(),
      ::mlir::pto::SyncOpType((*_result_op_type)));
}

void PipeEventTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifySyncOpType(getOpType());
  }
  odsPrinter << ">";
}

::mlir::pto::SyncOpType PipeEventTypeAttr::getOpType() const {
  return getImpl()->op_type;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::PipeEventTypeAttr)
namespace mlir {
namespace pto {
namespace detail {
struct SyncOpTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::pto::SyncOpType>;
  SyncOpTypeAttrStorage(::mlir::pto::SyncOpType op_type) : op_type(std::move(op_type)) {}

  KeyTy getAsKey() const {
    return KeyTy(op_type);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (op_type == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static SyncOpTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto op_type = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<SyncOpTypeAttrStorage>()) SyncOpTypeAttrStorage(std::move(op_type));
  }

  ::mlir::pto::SyncOpType op_type;
};
} // namespace detail
SyncOpTypeAttr SyncOpTypeAttr::get(::mlir::MLIRContext *context, ::mlir::pto::SyncOpType op_type) {
  return Base::get(context, std::move(op_type));
}

::mlir::Attribute SyncOpTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::pto::SyncOpType> _result_op_type;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'op_type'
  _result_op_type = [&]() -> ::mlir::FailureOr<::mlir::pto::SyncOpType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::pto::symbolizeSyncOpType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::pto::SyncOpType" << " to be one of: " << "TLOAD" << ", " << "TSTORE_ACC" << ", " << "TSTORE_VEC" << ", " << "TMOV_M2L" << ", " << "TMOV_M2S" << ", " << "TMOV_M2B" << ", " << "TMOV_M2V" << ", " << "TMOV_V2M" << ", " << "TMATMUL" << ", " << "TVEC" << ", " << "TVECWAIT_EVENT")};
    }();
  if (::mlir::failed(_result_op_type)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PTO_SyncOpTypeAttr parameter 'op_type' which is to be a `::mlir::pto::SyncOpType`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_op_type));
  return SyncOpTypeAttr::get(odsParser.getContext(),
      ::mlir::pto::SyncOpType((*_result_op_type)));
}

void SyncOpTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifySyncOpType(getOpType());
  }
  odsPrinter << ">";
}

::mlir::pto::SyncOpType SyncOpTypeAttr::getOpType() const {
  return getImpl()->op_type;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::SyncOpTypeAttr)
namespace mlir {
namespace pto {
namespace detail {
struct LayoutAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::pto::Layout>;
  LayoutAttrStorage(::mlir::pto::Layout layout) : layout(std::move(layout)) {}

  KeyTy getAsKey() const {
    return KeyTy(layout);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (layout == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static LayoutAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto layout = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<LayoutAttrStorage>()) LayoutAttrStorage(std::move(layout));
  }

  ::mlir::pto::Layout layout;
};
} // namespace detail
LayoutAttr LayoutAttr::get(::mlir::MLIRContext *context, ::mlir::pto::Layout layout) {
  return Base::get(context, std::move(layout));
}

::mlir::Attribute LayoutAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::pto::Layout> _result_layout;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'layout'
  _result_layout = [&]() -> ::mlir::FailureOr<::mlir::pto::Layout> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::pto::symbolizeLayout(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::pto::Layout" << " to be one of: " << "nd" << ", " << "dn" << ", " << "nz")};
    }();
  if (::mlir::failed(_result_layout)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PTO_LayoutAttr parameter 'layout' which is to be a `::mlir::pto::Layout`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_layout));
  return LayoutAttr::get(odsParser.getContext(),
      ::mlir::pto::Layout((*_result_layout)));
}

void LayoutAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyLayout(getLayout());
  }
  odsPrinter << ">";
}

::mlir::pto::Layout LayoutAttr::getLayout() const {
  return getImpl()->layout;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::LayoutAttr)
namespace mlir {
namespace pto {
namespace detail {
struct TFuncCoreTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::pto::TFuncCoreType>;
  TFuncCoreTypeAttrStorage(::mlir::pto::TFuncCoreType funcCoreType) : funcCoreType(std::move(funcCoreType)) {}

  KeyTy getAsKey() const {
    return KeyTy(funcCoreType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (funcCoreType == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TFuncCoreTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto funcCoreType = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<TFuncCoreTypeAttrStorage>()) TFuncCoreTypeAttrStorage(std::move(funcCoreType));
  }

  ::mlir::pto::TFuncCoreType funcCoreType;
};
} // namespace detail
TFuncCoreTypeAttr TFuncCoreTypeAttr::get(::mlir::MLIRContext *context, ::mlir::pto::TFuncCoreType funcCoreType) {
  return Base::get(context, std::move(funcCoreType));
}

::mlir::Attribute TFuncCoreTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::pto::TFuncCoreType> _result_funcCoreType;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'funcCoreType'
  _result_funcCoreType = [&]() -> ::mlir::FailureOr<::mlir::pto::TFuncCoreType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::pto::symbolizeTFuncCoreType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::pto::TFuncCoreType" << " to be one of: " << "AIC" << ", " << "AIV" << ", " << "MIX" << ", " << "AIC_OR_AIV")};
    }();
  if (::mlir::failed(_result_funcCoreType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PTO_TFuncCoreTypeAttr parameter 'funcCoreType' which is to be a `::mlir::pto::TFuncCoreType`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_funcCoreType));
  return TFuncCoreTypeAttr::get(odsParser.getContext(),
      ::mlir::pto::TFuncCoreType((*_result_funcCoreType)));
}

void TFuncCoreTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyTFuncCoreType(getFuncCoreType());
  }
  odsPrinter << ">";
}

::mlir::pto::TFuncCoreType TFuncCoreTypeAttr::getFuncCoreType() const {
  return getImpl()->funcCoreType;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::TFuncCoreTypeAttr)
namespace mlir {
namespace pto {
namespace detail {
struct TModuleCoreTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::pto::TModuleCoreType>;
  TModuleCoreTypeAttrStorage(::mlir::pto::TModuleCoreType moduleCoreType) : moduleCoreType(std::move(moduleCoreType)) {}

  KeyTy getAsKey() const {
    return KeyTy(moduleCoreType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (moduleCoreType == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TModuleCoreTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto moduleCoreType = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<TModuleCoreTypeAttrStorage>()) TModuleCoreTypeAttrStorage(std::move(moduleCoreType));
  }

  ::mlir::pto::TModuleCoreType moduleCoreType;
};
} // namespace detail
TModuleCoreTypeAttr TModuleCoreTypeAttr::get(::mlir::MLIRContext *context, ::mlir::pto::TModuleCoreType moduleCoreType) {
  return Base::get(context, std::move(moduleCoreType));
}

::mlir::Attribute TModuleCoreTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::pto::TModuleCoreType> _result_moduleCoreType;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'moduleCoreType'
  _result_moduleCoreType = [&]() -> ::mlir::FailureOr<::mlir::pto::TModuleCoreType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::pto::symbolizeTModuleCoreType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::pto::TModuleCoreType" << " to be one of: " << "AIC" << ", " << "AIV" << ", " << "MIX")};
    }();
  if (::mlir::failed(_result_moduleCoreType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PTO_TModuleCoreTypeAttr parameter 'moduleCoreType' which is to be a `::mlir::pto::TModuleCoreType`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_moduleCoreType));
  return TModuleCoreTypeAttr::get(odsParser.getContext(),
      ::mlir::pto::TModuleCoreType((*_result_moduleCoreType)));
}

void TModuleCoreTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyTModuleCoreType(getModuleCoreType());
  }
  odsPrinter << ">";
}

::mlir::pto::TModuleCoreType TModuleCoreTypeAttr::getModuleCoreType() const {
  return getImpl()->moduleCoreType;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::TModuleCoreTypeAttr)
namespace mlir {
namespace pto {
namespace detail {
struct EventAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::pto::EVENT>;
  EventAttrStorage(::mlir::pto::EVENT event) : event(std::move(event)) {}

  KeyTy getAsKey() const {
    return KeyTy(event);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (event == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static EventAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto event = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<EventAttrStorage>()) EventAttrStorage(std::move(event));
  }

  ::mlir::pto::EVENT event;
};
} // namespace detail
EventAttr EventAttr::get(::mlir::MLIRContext *context, ::mlir::pto::EVENT event) {
  return Base::get(context, std::move(event));
}

::mlir::Attribute EventAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::pto::EVENT> _result_event;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'event'
  _result_event = [&]() -> ::mlir::FailureOr<::mlir::pto::EVENT> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::pto::symbolizeEVENT(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::pto::EVENT" << " to be one of: " << "EVENT_ID0" << ", " << "EVENT_ID1" << ", " << "EVENT_ID2" << ", " << "EVENT_ID3" << ", " << "EVENT_ID4" << ", " << "EVENT_ID5" << ", " << "EVENT_ID6" << ", " << "EVENT_ID7")};
    }();
  if (::mlir::failed(_result_event)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PTO_EventAttr parameter 'event' which is to be a `::mlir::pto::EVENT`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_event));
  return EventAttr::get(odsParser.getContext(),
      ::mlir::pto::EVENT((*_result_event)));
}

void EventAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyEVENT(getEvent());
  }
  odsPrinter << ">";
}

::mlir::pto::EVENT EventAttr::getEvent() const {
  return getImpl()->event;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::EventAttr)
namespace mlir {
namespace pto {
namespace detail {
struct PadModeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::pto::PadMode>;
  PadModeAttrStorage(::mlir::pto::PadMode padmode) : padmode(std::move(padmode)) {}

  KeyTy getAsKey() const {
    return KeyTy(padmode);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (padmode == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static PadModeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto padmode = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<PadModeAttrStorage>()) PadModeAttrStorage(std::move(padmode));
  }

  ::mlir::pto::PadMode padmode;
};
} // namespace detail
PadModeAttr PadModeAttr::get(::mlir::MLIRContext *context, ::mlir::pto::PadMode padmode) {
  return Base::get(context, std::move(padmode));
}

::mlir::Attribute PadModeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::pto::PadMode> _result_padmode;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'padmode'
  _result_padmode = [&]() -> ::mlir::FailureOr<::mlir::pto::PadMode> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::pto::symbolizePadMode(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::pto::PadMode" << " to be one of: " << "PadNull" << ", " << "PadFirstElem" << ", " << "PadValue")};
    }();
  if (::mlir::failed(_result_padmode)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PTO_PadModeAttr parameter 'padmode' which is to be a `::mlir::pto::PadMode`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_padmode));
  return PadModeAttr::get(odsParser.getContext(),
      ::mlir::pto::PadMode((*_result_padmode)));
}

void PadModeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyPadMode(getPadmode());
  }
  odsPrinter << ">";
}

::mlir::pto::PadMode PadModeAttr::getPadmode() const {
  return getImpl()->padmode;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::PadModeAttr)
namespace mlir {
namespace pto {
} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::VectorFunctionAttr)
namespace mlir {
namespace pto {
namespace detail {
struct CmpModeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::pto::CmpMode>;
  CmpModeAttrStorage(::mlir::pto::CmpMode value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static CmpModeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<CmpModeAttrStorage>()) CmpModeAttrStorage(std::move(value));
  }

  ::mlir::pto::CmpMode value;
};
} // namespace detail
CmpModeAttr CmpModeAttr::get(::mlir::MLIRContext *context, ::mlir::pto::CmpMode value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute CmpModeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::pto::CmpMode> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::pto::CmpMode> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::pto::symbolizeCmpMode(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::pto::CmpMode" << " to be one of: " << "eq" << ", " << "ne" << ", " << "lt" << ", " << "le" << ", " << "gt" << ", " << "ge")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PTO_CmpModeAttr parameter 'value' which is to be a `::mlir::pto::CmpMode`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return CmpModeAttr::get(odsParser.getContext(),
      ::mlir::pto::CmpMode((*_result_value)));
}

void CmpModeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyCmpMode(getValue());
}

::mlir::pto::CmpMode CmpModeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::CmpModeAttr)
namespace mlir {
namespace pto {
namespace detail {
struct RoundModeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::pto::RoundMode>;
  RoundModeAttrStorage(::mlir::pto::RoundMode value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static RoundModeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<RoundModeAttrStorage>()) RoundModeAttrStorage(std::move(value));
  }

  ::mlir::pto::RoundMode value;
};
} // namespace detail
RoundModeAttr RoundModeAttr::get(::mlir::MLIRContext *context, ::mlir::pto::RoundMode value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute RoundModeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::pto::RoundMode> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::pto::RoundMode> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::pto::symbolizeRoundMode(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::pto::RoundMode" << " to be one of: " << "NONE" << ", " << "RINT" << ", " << "ROUND" << ", " << "FLOOR" << ", " << "CEIL" << ", " << "TRUNC" << ", " << "ODD" << ", " << "CAST_RINT")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PTO_RoundModeAttr parameter 'value' which is to be a `::mlir::pto::RoundMode`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return RoundModeAttr::get(odsParser.getContext(),
      ::mlir::pto::RoundMode((*_result_value)));
}

void RoundModeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyRoundMode(getValue());
}

::mlir::pto::RoundMode RoundModeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::RoundModeAttr)
namespace mlir {
namespace pto {
namespace detail {
struct MaskPatternAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::pto::MaskPattern>;
  MaskPatternAttrStorage(::mlir::pto::MaskPattern value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static MaskPatternAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<MaskPatternAttrStorage>()) MaskPatternAttrStorage(std::move(value));
  }

  ::mlir::pto::MaskPattern value;
};
} // namespace detail
MaskPatternAttr MaskPatternAttr::get(::mlir::MLIRContext *context, ::mlir::pto::MaskPattern value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute MaskPatternAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::pto::MaskPattern> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::pto::MaskPattern> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::pto::symbolizeMaskPattern(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::pto::MaskPattern" << " to be one of: " << "P0101" << ", " << "P0011" << ", " << "P0110" << ", " << "P0001" << ", " << "P1111" << ", " << "P1010")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PTO_MaskPatternAttr parameter 'value' which is to be a `::mlir::pto::MaskPattern`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return MaskPatternAttr::get(odsParser.getContext(),
      ::mlir::pto::MaskPattern((*_result_value)));
}

void MaskPatternAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyMaskPattern(getValue());
  }
  odsPrinter << ">";
}

::mlir::pto::MaskPattern MaskPatternAttr::getValue() const {
  return getImpl()->value;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::MaskPatternAttr)
namespace mlir {
namespace pto {
namespace detail {
struct BLayoutAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::pto::BLayout>;
  BLayoutAttrStorage(::mlir::pto::BLayout value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static BLayoutAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<BLayoutAttrStorage>()) BLayoutAttrStorage(std::move(value));
  }

  ::mlir::pto::BLayout value;
};
} // namespace detail
BLayoutAttr BLayoutAttr::get(::mlir::MLIRContext *context, ::mlir::pto::BLayout value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute BLayoutAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::pto::BLayout> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::pto::BLayout> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::pto::symbolizeBLayout(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::pto::BLayout" << " to be one of: " << "row_major" << ", " << "col_major")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PTO_BLayoutAttr parameter 'value' which is to be a `::mlir::pto::BLayout`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return BLayoutAttr::get(odsParser.getContext(),
      ::mlir::pto::BLayout((*_result_value)));
}

void BLayoutAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyBLayout(getValue());
  }
  odsPrinter << ">";
}

::mlir::pto::BLayout BLayoutAttr::getValue() const {
  return getImpl()->value;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::BLayoutAttr)
namespace mlir {
namespace pto {
namespace detail {
struct SLayoutAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::pto::SLayout>;
  SLayoutAttrStorage(::mlir::pto::SLayout value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static SLayoutAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<SLayoutAttrStorage>()) SLayoutAttrStorage(std::move(value));
  }

  ::mlir::pto::SLayout value;
};
} // namespace detail
SLayoutAttr SLayoutAttr::get(::mlir::MLIRContext *context, ::mlir::pto::SLayout value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute SLayoutAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::pto::SLayout> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::pto::SLayout> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::pto::symbolizeSLayout(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::pto::SLayout" << " to be one of: " << "none_box" << ", " << "row_major" << ", " << "col_major")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PTO_SLayoutAttr parameter 'value' which is to be a `::mlir::pto::SLayout`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return SLayoutAttr::get(odsParser.getContext(),
      ::mlir::pto::SLayout((*_result_value)));
}

void SLayoutAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifySLayout(getValue());
  }
  odsPrinter << ">";
}

::mlir::pto::SLayout SLayoutAttr::getValue() const {
  return getImpl()->value;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::SLayoutAttr)
namespace mlir {
namespace pto {
namespace detail {
struct PadValueAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::pto::PadValue>;
  PadValueAttrStorage(::mlir::pto::PadValue value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static PadValueAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<PadValueAttrStorage>()) PadValueAttrStorage(std::move(value));
  }

  ::mlir::pto::PadValue value;
};
} // namespace detail
PadValueAttr PadValueAttr::get(::mlir::MLIRContext *context, ::mlir::pto::PadValue value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute PadValueAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::pto::PadValue> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter list

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::pto::PadValue> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::pto::symbolizePadValue(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::pto::PadValue" << " to be one of: " << "null" << ", " << "zero" << ", " << "max" << ", " << "min")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PTO_PadValueAttr parameter 'value' which is to be a `::mlir::pto::PadValue`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return PadValueAttr::get(odsParser.getContext(),
      ::mlir::pto::PadValue((*_result_value)));
}

void PadValueAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << stringifyPadValue(getValue());
  }
  odsPrinter << ">";
}

::mlir::pto::PadValue PadValueAttr::getValue() const {
  return getImpl()->value;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::PadValueAttr)
namespace mlir {
namespace pto {
namespace detail {
struct TileBufConfigAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<BLayoutAttr, SLayoutAttr, mlir::IntegerAttr, PadValueAttr>;
  TileBufConfigAttrStorage(BLayoutAttr bLayout, SLayoutAttr sLayout, mlir::IntegerAttr sFractalSize, PadValueAttr pad) : bLayout(std::move(bLayout)), sLayout(std::move(sLayout)), sFractalSize(std::move(sFractalSize)), pad(std::move(pad)) {}

  KeyTy getAsKey() const {
    return KeyTy(bLayout, sLayout, sFractalSize, pad);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (bLayout == std::get<0>(tblgenKey)) && (sLayout == std::get<1>(tblgenKey)) && (sFractalSize == std::get<2>(tblgenKey)) && (pad == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static TileBufConfigAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto bLayout = std::move(std::get<0>(tblgenKey));
    auto sLayout = std::move(std::get<1>(tblgenKey));
    auto sFractalSize = std::move(std::get<2>(tblgenKey));
    auto pad = std::move(std::get<3>(tblgenKey));
    return new (allocator.allocate<TileBufConfigAttrStorage>()) TileBufConfigAttrStorage(std::move(bLayout), std::move(sLayout), std::move(sFractalSize), std::move(pad));
  }

  BLayoutAttr bLayout;
  SLayoutAttr sLayout;
  mlir::IntegerAttr sFractalSize;
  PadValueAttr pad;
};
} // namespace detail
TileBufConfigAttr TileBufConfigAttr::get(::mlir::MLIRContext *context, BLayoutAttr bLayout, SLayoutAttr sLayout, mlir::IntegerAttr sFractalSize, PadValueAttr pad) {
  return Base::get(context, std::move(bLayout), std::move(sLayout), std::move(sFractalSize), std::move(pad));
}

BLayoutAttr TileBufConfigAttr::getBLayout() const {
  return getImpl()->bLayout;
}

SLayoutAttr TileBufConfigAttr::getSLayout() const {
  return getImpl()->sLayout;
}

mlir::IntegerAttr TileBufConfigAttr::getSFractalSize() const {
  return getImpl()->sFractalSize;
}

PadValueAttr TileBufConfigAttr::getPad() const {
  return getImpl()->pad;
}

} // namespace pto
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pto::TileBufConfigAttr)
namespace mlir {
namespace pto {

/// Parse an attribute registered to this dialect.
::mlir::Attribute PTODialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                      ::mlir::Type type) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef attrTag;
  {
    ::mlir::Attribute attr;
    auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
    if (parseResult.has_value())
      return attr;
  }
  
  parser.emitError(typeLoc) << "unknown attribute `"
      << attrTag << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print an attribute registered to this dialect.
void PTODialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace pto
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES


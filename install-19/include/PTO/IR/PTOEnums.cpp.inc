/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: PTOOps.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace pto {
::llvm::StringRef stringifyAddressSpace(AddressSpace val) {
  switch (val) {
    case AddressSpace::Zero: return "zero";
    case AddressSpace::GM: return "gm";
    case AddressSpace::MAT: return "mat";
    case AddressSpace::LEFT: return "left";
    case AddressSpace::RIGHT: return "right";
    case AddressSpace::ACC: return "acc";
    case AddressSpace::VEC: return "vec";
    case AddressSpace::BIAS: return "bias";
    case AddressSpace::SCALING: return "scaling";
  }
  return "";
}

::std::optional<AddressSpace> symbolizeAddressSpace(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<AddressSpace>>(str)
      .Case("zero", AddressSpace::Zero)
      .Case("gm", AddressSpace::GM)
      .Case("mat", AddressSpace::MAT)
      .Case("left", AddressSpace::LEFT)
      .Case("right", AddressSpace::RIGHT)
      .Case("acc", AddressSpace::ACC)
      .Case("vec", AddressSpace::VEC)
      .Case("bias", AddressSpace::BIAS)
      .Case("scaling", AddressSpace::SCALING)
      .Default(::std::nullopt);
}
::std::optional<AddressSpace> symbolizeAddressSpace(uint32_t value) {
  switch (value) {
  case 0: return AddressSpace::Zero;
  case 1: return AddressSpace::GM;
  case 2: return AddressSpace::MAT;
  case 3: return AddressSpace::LEFT;
  case 4: return AddressSpace::RIGHT;
  case 5: return AddressSpace::ACC;
  case 6: return AddressSpace::VEC;
  case 7: return AddressSpace::BIAS;
  case 8: return AddressSpace::SCALING;
  default: return ::std::nullopt;
  }
}

} // namespace pto
} // namespace mlir

namespace mlir {
namespace pto {
::llvm::StringRef stringifyBLayout(BLayout val) {
  switch (val) {
    case BLayout::RowMajor: return "row_major";
    case BLayout::ColMajor: return "col_major";
  }
  return "";
}

::std::optional<BLayout> symbolizeBLayout(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<BLayout>>(str)
      .Case("row_major", BLayout::RowMajor)
      .Case("col_major", BLayout::ColMajor)
      .Default(::std::nullopt);
}
::std::optional<BLayout> symbolizeBLayout(uint32_t value) {
  switch (value) {
  case 0: return BLayout::RowMajor;
  case 1: return BLayout::ColMajor;
  default: return ::std::nullopt;
  }
}

} // namespace pto
} // namespace mlir

namespace mlir {
namespace pto {
::llvm::StringRef stringifyCmpMode(CmpMode val) {
  switch (val) {
    case CmpMode::EQ: return "eq";
    case CmpMode::NE: return "ne";
    case CmpMode::LT: return "lt";
    case CmpMode::LE: return "le";
    case CmpMode::GT: return "gt";
    case CmpMode::GE: return "ge";
  }
  return "";
}

::std::optional<CmpMode> symbolizeCmpMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<CmpMode>>(str)
      .Case("eq", CmpMode::EQ)
      .Case("ne", CmpMode::NE)
      .Case("lt", CmpMode::LT)
      .Case("le", CmpMode::LE)
      .Case("gt", CmpMode::GT)
      .Case("ge", CmpMode::GE)
      .Default(::std::nullopt);
}
::std::optional<CmpMode> symbolizeCmpMode(uint32_t value) {
  switch (value) {
  case 0: return CmpMode::EQ;
  case 1: return CmpMode::NE;
  case 2: return CmpMode::LT;
  case 3: return CmpMode::LE;
  case 4: return CmpMode::GT;
  case 5: return CmpMode::GE;
  default: return ::std::nullopt;
  }
}

} // namespace pto
} // namespace mlir

namespace mlir {
namespace pto {
::llvm::StringRef stringifyEVENT(EVENT val) {
  switch (val) {
    case EVENT::EVENT_ID0: return "EVENT_ID0";
    case EVENT::EVENT_ID1: return "EVENT_ID1";
    case EVENT::EVENT_ID2: return "EVENT_ID2";
    case EVENT::EVENT_ID3: return "EVENT_ID3";
    case EVENT::EVENT_ID4: return "EVENT_ID4";
    case EVENT::EVENT_ID5: return "EVENT_ID5";
    case EVENT::EVENT_ID6: return "EVENT_ID6";
    case EVENT::EVENT_ID7: return "EVENT_ID7";
  }
  return "";
}

::std::optional<EVENT> symbolizeEVENT(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<EVENT>>(str)
      .Case("EVENT_ID0", EVENT::EVENT_ID0)
      .Case("EVENT_ID1", EVENT::EVENT_ID1)
      .Case("EVENT_ID2", EVENT::EVENT_ID2)
      .Case("EVENT_ID3", EVENT::EVENT_ID3)
      .Case("EVENT_ID4", EVENT::EVENT_ID4)
      .Case("EVENT_ID5", EVENT::EVENT_ID5)
      .Case("EVENT_ID6", EVENT::EVENT_ID6)
      .Case("EVENT_ID7", EVENT::EVENT_ID7)
      .Default(::std::nullopt);
}
::std::optional<EVENT> symbolizeEVENT(uint32_t value) {
  switch (value) {
  case 0: return EVENT::EVENT_ID0;
  case 1: return EVENT::EVENT_ID1;
  case 2: return EVENT::EVENT_ID2;
  case 3: return EVENT::EVENT_ID3;
  case 4: return EVENT::EVENT_ID4;
  case 5: return EVENT::EVENT_ID5;
  case 6: return EVENT::EVENT_ID6;
  case 7: return EVENT::EVENT_ID7;
  default: return ::std::nullopt;
  }
}

} // namespace pto
} // namespace mlir

namespace mlir {
namespace pto {
::llvm::StringRef stringifyLayout(Layout val) {
  switch (val) {
    case Layout::ND: return "nd";
    case Layout::DN: return "dn";
    case Layout::NZ: return "nz";
  }
  return "";
}

::std::optional<Layout> symbolizeLayout(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Layout>>(str)
      .Case("nd", Layout::ND)
      .Case("dn", Layout::DN)
      .Case("nz", Layout::NZ)
      .Default(::std::nullopt);
}
::std::optional<Layout> symbolizeLayout(uint32_t value) {
  switch (value) {
  case 0: return Layout::ND;
  case 1: return Layout::DN;
  case 2: return Layout::NZ;
  default: return ::std::nullopt;
  }
}

} // namespace pto
} // namespace mlir

namespace mlir {
namespace pto {
::llvm::StringRef stringifyMaskPattern(MaskPattern val) {
  switch (val) {
    case MaskPattern::P0101: return "P0101";
    case MaskPattern::P0011: return "P0011";
    case MaskPattern::P0110: return "P0110";
    case MaskPattern::P0001: return "P0001";
    case MaskPattern::P1111: return "P1111";
    case MaskPattern::P1010: return "P1010";
  }
  return "";
}

::std::optional<MaskPattern> symbolizeMaskPattern(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<MaskPattern>>(str)
      .Case("P0101", MaskPattern::P0101)
      .Case("P0011", MaskPattern::P0011)
      .Case("P0110", MaskPattern::P0110)
      .Case("P0001", MaskPattern::P0001)
      .Case("P1111", MaskPattern::P1111)
      .Case("P1010", MaskPattern::P1010)
      .Default(::std::nullopt);
}
::std::optional<MaskPattern> symbolizeMaskPattern(uint32_t value) {
  switch (value) {
  case 0: return MaskPattern::P0101;
  case 1: return MaskPattern::P0011;
  case 2: return MaskPattern::P0110;
  case 3: return MaskPattern::P0001;
  case 4: return MaskPattern::P1111;
  case 5: return MaskPattern::P1010;
  default: return ::std::nullopt;
  }
}

} // namespace pto
} // namespace mlir

namespace mlir {
namespace pto {
::llvm::StringRef stringifyMemPlanMode(MemPlanMode val) {
  switch (val) {
    case MemPlanMode::LOCAL_MEM_PLAN: return "LOCAL_MEM_PLAN";
    case MemPlanMode::GLOBAL_WORKSPACE_PLAN: return "GLOBAL_WORKSPACE_PLAN";
  }
  return "";
}

::std::optional<MemPlanMode> symbolizeMemPlanMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<MemPlanMode>>(str)
      .Case("LOCAL_MEM_PLAN", MemPlanMode::LOCAL_MEM_PLAN)
      .Case("GLOBAL_WORKSPACE_PLAN", MemPlanMode::GLOBAL_WORKSPACE_PLAN)
      .Default(::std::nullopt);
}
::std::optional<MemPlanMode> symbolizeMemPlanMode(uint32_t value) {
  switch (value) {
  case 0: return MemPlanMode::LOCAL_MEM_PLAN;
  case 1: return MemPlanMode::GLOBAL_WORKSPACE_PLAN;
  default: return ::std::nullopt;
  }
}

} // namespace pto
} // namespace mlir

namespace mlir {
namespace pto {
::llvm::StringRef stringifyPadMode(PadMode val) {
  switch (val) {
    case PadMode::PadNull: return "PadNull";
    case PadMode::PadFirstElem: return "PadFirstElem";
    case PadMode::PadValue: return "PadValue";
  }
  return "";
}

::std::optional<PadMode> symbolizePadMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<PadMode>>(str)
      .Case("PadNull", PadMode::PadNull)
      .Case("PadFirstElem", PadMode::PadFirstElem)
      .Case("PadValue", PadMode::PadValue)
      .Default(::std::nullopt);
}
::std::optional<PadMode> symbolizePadMode(uint32_t value) {
  switch (value) {
  case 0: return PadMode::PadNull;
  case 1: return PadMode::PadFirstElem;
  case 2: return PadMode::PadValue;
  default: return ::std::nullopt;
  }
}

} // namespace pto
} // namespace mlir

namespace mlir {
namespace pto {
::llvm::StringRef stringifyPadValue(PadValue val) {
  switch (val) {
    case PadValue::Null: return "null";
    case PadValue::Zero: return "zero";
    case PadValue::Max: return "max";
    case PadValue::Min: return "min";
  }
  return "";
}

::std::optional<PadValue> symbolizePadValue(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<PadValue>>(str)
      .Case("null", PadValue::Null)
      .Case("zero", PadValue::Zero)
      .Case("max", PadValue::Max)
      .Case("min", PadValue::Min)
      .Default(::std::nullopt);
}
::std::optional<PadValue> symbolizePadValue(uint32_t value) {
  switch (value) {
  case 0: return PadValue::Null;
  case 1: return PadValue::Zero;
  case 2: return PadValue::Max;
  case 3: return PadValue::Min;
  default: return ::std::nullopt;
  }
}

} // namespace pto
} // namespace mlir

namespace mlir {
namespace pto {
::llvm::StringRef stringifyPIPE(PIPE val) {
  switch (val) {
    case PIPE::PIPE_S: return "PIPE_S";
    case PIPE::PIPE_V: return "PIPE_V";
    case PIPE::PIPE_M: return "PIPE_M";
    case PIPE::PIPE_MTE1: return "PIPE_MTE1";
    case PIPE::PIPE_MTE2: return "PIPE_MTE2";
    case PIPE::PIPE_MTE3: return "PIPE_MTE3";
    case PIPE::PIPE_ALL: return "PIPE_ALL";
    case PIPE::PIPE_MTE4: return "PIPE_MTE4";
    case PIPE::PIPE_MTE5: return "PIPE_MTE5";
    case PIPE::PIPE_V2: return "PIPE_V2";
    case PIPE::PIPE_FIX: return "PIPE_FIX";
    case PIPE::VIRTUAL_PIPE_MTE2_L1A: return "VIRTUAL_PIPE_MTE2_L1A";
    case PIPE::VIRTUAL_PIPE_MTE2_L1B: return "VIRTUAL_PIPE_MTE2_L1B";
    case PIPE::PIPE_NUM: return "PIPE_NUM";
    case PIPE::PIPE_UNASSIGNED: return "PIPE_UNASSIGNED";
  }
  return "";
}

::std::optional<PIPE> symbolizePIPE(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<PIPE>>(str)
      .Case("PIPE_S", PIPE::PIPE_S)
      .Case("PIPE_V", PIPE::PIPE_V)
      .Case("PIPE_M", PIPE::PIPE_M)
      .Case("PIPE_MTE1", PIPE::PIPE_MTE1)
      .Case("PIPE_MTE2", PIPE::PIPE_MTE2)
      .Case("PIPE_MTE3", PIPE::PIPE_MTE3)
      .Case("PIPE_ALL", PIPE::PIPE_ALL)
      .Case("PIPE_MTE4", PIPE::PIPE_MTE4)
      .Case("PIPE_MTE5", PIPE::PIPE_MTE5)
      .Case("PIPE_V2", PIPE::PIPE_V2)
      .Case("PIPE_FIX", PIPE::PIPE_FIX)
      .Case("VIRTUAL_PIPE_MTE2_L1A", PIPE::VIRTUAL_PIPE_MTE2_L1A)
      .Case("VIRTUAL_PIPE_MTE2_L1B", PIPE::VIRTUAL_PIPE_MTE2_L1B)
      .Case("PIPE_NUM", PIPE::PIPE_NUM)
      .Case("PIPE_UNASSIGNED", PIPE::PIPE_UNASSIGNED)
      .Default(::std::nullopt);
}
::std::optional<PIPE> symbolizePIPE(uint32_t value) {
  switch (value) {
  case 0: return PIPE::PIPE_S;
  case 1: return PIPE::PIPE_V;
  case 2: return PIPE::PIPE_M;
  case 3: return PIPE::PIPE_MTE1;
  case 4: return PIPE::PIPE_MTE2;
  case 5: return PIPE::PIPE_MTE3;
  case 6: return PIPE::PIPE_ALL;
  case 7: return PIPE::PIPE_MTE4;
  case 8: return PIPE::PIPE_MTE5;
  case 9: return PIPE::PIPE_V2;
  case 10: return PIPE::PIPE_FIX;
  case 11: return PIPE::VIRTUAL_PIPE_MTE2_L1A;
  case 12: return PIPE::VIRTUAL_PIPE_MTE2_L1B;
  case 13: return PIPE::PIPE_NUM;
  case 99: return PIPE::PIPE_UNASSIGNED;
  default: return ::std::nullopt;
  }
}

} // namespace pto
} // namespace mlir

namespace mlir {
namespace pto {
::llvm::StringRef stringifyRoundMode(RoundMode val) {
  switch (val) {
    case RoundMode::NONE: return "NONE";
    case RoundMode::RINT: return "RINT";
    case RoundMode::ROUND: return "ROUND";
    case RoundMode::FLOOR: return "FLOOR";
    case RoundMode::CEIL: return "CEIL";
    case RoundMode::TRUNC: return "TRUNC";
    case RoundMode::ODD: return "ODD";
    case RoundMode::CAST_RINT: return "CAST_RINT";
  }
  return "";
}

::std::optional<RoundMode> symbolizeRoundMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<RoundMode>>(str)
      .Case("NONE", RoundMode::NONE)
      .Case("RINT", RoundMode::RINT)
      .Case("ROUND", RoundMode::ROUND)
      .Case("FLOOR", RoundMode::FLOOR)
      .Case("CEIL", RoundMode::CEIL)
      .Case("TRUNC", RoundMode::TRUNC)
      .Case("ODD", RoundMode::ODD)
      .Case("CAST_RINT", RoundMode::CAST_RINT)
      .Default(::std::nullopt);
}
::std::optional<RoundMode> symbolizeRoundMode(uint32_t value) {
  switch (value) {
  case 0: return RoundMode::NONE;
  case 1: return RoundMode::RINT;
  case 2: return RoundMode::ROUND;
  case 3: return RoundMode::FLOOR;
  case 4: return RoundMode::CEIL;
  case 5: return RoundMode::TRUNC;
  case 6: return RoundMode::ODD;
  case 7: return RoundMode::CAST_RINT;
  default: return ::std::nullopt;
  }
}

} // namespace pto
} // namespace mlir

namespace mlir {
namespace pto {
::llvm::StringRef stringifySLayout(SLayout val) {
  switch (val) {
    case SLayout::NoneBox: return "none_box";
    case SLayout::RowMajor: return "row_major";
    case SLayout::ColMajor: return "col_major";
  }
  return "";
}

::std::optional<SLayout> symbolizeSLayout(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<SLayout>>(str)
      .Case("none_box", SLayout::NoneBox)
      .Case("row_major", SLayout::RowMajor)
      .Case("col_major", SLayout::ColMajor)
      .Default(::std::nullopt);
}
::std::optional<SLayout> symbolizeSLayout(uint32_t value) {
  switch (value) {
  case 0: return SLayout::NoneBox;
  case 1: return SLayout::RowMajor;
  case 2: return SLayout::ColMajor;
  default: return ::std::nullopt;
  }
}

} // namespace pto
} // namespace mlir

namespace mlir {
namespace pto {
::llvm::StringRef stringifySyncOpType(SyncOpType val) {
  switch (val) {
    case SyncOpType::TLOAD: return "TLOAD";
    case SyncOpType::TSTORE_ACC: return "TSTORE_ACC";
    case SyncOpType::TSTORE_VEC: return "TSTORE_VEC";
    case SyncOpType::TMOV_M2L: return "TMOV_M2L";
    case SyncOpType::TMOV_M2S: return "TMOV_M2S";
    case SyncOpType::TMOV_M2B: return "TMOV_M2B";
    case SyncOpType::TMOV_M2V: return "TMOV_M2V";
    case SyncOpType::TMOV_V2M: return "TMOV_V2M";
    case SyncOpType::TMATMUL: return "TMATMUL";
    case SyncOpType::TVEC: return "TVEC";
    case SyncOpType::TVECWAIT_EVENT: return "TVECWAIT_EVENT";
  }
  return "";
}

::std::optional<SyncOpType> symbolizeSyncOpType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<SyncOpType>>(str)
      .Case("TLOAD", SyncOpType::TLOAD)
      .Case("TSTORE_ACC", SyncOpType::TSTORE_ACC)
      .Case("TSTORE_VEC", SyncOpType::TSTORE_VEC)
      .Case("TMOV_M2L", SyncOpType::TMOV_M2L)
      .Case("TMOV_M2S", SyncOpType::TMOV_M2S)
      .Case("TMOV_M2B", SyncOpType::TMOV_M2B)
      .Case("TMOV_M2V", SyncOpType::TMOV_M2V)
      .Case("TMOV_V2M", SyncOpType::TMOV_V2M)
      .Case("TMATMUL", SyncOpType::TMATMUL)
      .Case("TVEC", SyncOpType::TVEC)
      .Case("TVECWAIT_EVENT", SyncOpType::TVECWAIT_EVENT)
      .Default(::std::nullopt);
}
::std::optional<SyncOpType> symbolizeSyncOpType(uint32_t value) {
  switch (value) {
  case 0: return SyncOpType::TLOAD;
  case 1: return SyncOpType::TSTORE_ACC;
  case 2: return SyncOpType::TSTORE_VEC;
  case 3: return SyncOpType::TMOV_M2L;
  case 4: return SyncOpType::TMOV_M2S;
  case 5: return SyncOpType::TMOV_M2B;
  case 6: return SyncOpType::TMOV_M2V;
  case 7: return SyncOpType::TMOV_V2M;
  case 8: return SyncOpType::TMATMUL;
  case 9: return SyncOpType::TVEC;
  case 10: return SyncOpType::TVECWAIT_EVENT;
  default: return ::std::nullopt;
  }
}

} // namespace pto
} // namespace mlir

namespace mlir {
namespace pto {
::llvm::StringRef stringifyTFuncCoreType(TFuncCoreType val) {
  switch (val) {
    case TFuncCoreType::AIC: return "AIC";
    case TFuncCoreType::AIV: return "AIV";
    case TFuncCoreType::MIX: return "MIX";
    case TFuncCoreType::AIC_OR_AIV: return "AIC_OR_AIV";
  }
  return "";
}

::std::optional<TFuncCoreType> symbolizeTFuncCoreType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<TFuncCoreType>>(str)
      .Case("AIC", TFuncCoreType::AIC)
      .Case("AIV", TFuncCoreType::AIV)
      .Case("MIX", TFuncCoreType::MIX)
      .Case("AIC_OR_AIV", TFuncCoreType::AIC_OR_AIV)
      .Default(::std::nullopt);
}
::std::optional<TFuncCoreType> symbolizeTFuncCoreType(uint32_t value) {
  switch (value) {
  case 1: return TFuncCoreType::AIC;
  case 2: return TFuncCoreType::AIV;
  case 3: return TFuncCoreType::MIX;
  case 4: return TFuncCoreType::AIC_OR_AIV;
  default: return ::std::nullopt;
  }
}

} // namespace pto
} // namespace mlir

namespace mlir {
namespace pto {
::llvm::StringRef stringifyTModuleCoreType(TModuleCoreType val) {
  switch (val) {
    case TModuleCoreType::AIC: return "AIC";
    case TModuleCoreType::AIV: return "AIV";
    case TModuleCoreType::MIX: return "MIX";
  }
  return "";
}

::std::optional<TModuleCoreType> symbolizeTModuleCoreType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<TModuleCoreType>>(str)
      .Case("AIC", TModuleCoreType::AIC)
      .Case("AIV", TModuleCoreType::AIV)
      .Case("MIX", TModuleCoreType::MIX)
      .Default(::std::nullopt);
}
::std::optional<TModuleCoreType> symbolizeTModuleCoreType(uint32_t value) {
  switch (value) {
  case 1: return TModuleCoreType::AIC;
  case 2: return TModuleCoreType::AIV;
  case 3: return TModuleCoreType::MIX;
  default: return ::std::nullopt;
  }
}

} // namespace pto
} // namespace mlir


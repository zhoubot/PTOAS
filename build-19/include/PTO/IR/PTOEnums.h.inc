/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: PTOOps.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace pto {
// PTO Address Space
enum class AddressSpace : uint32_t {
  Zero = 0,
  GM = 1,
  MAT = 2,
  LEFT = 3,
  RIGHT = 4,
  ACC = 5,
  VEC = 6,
  BIAS = 7,
  SCALING = 8,
};

::std::optional<AddressSpace> symbolizeAddressSpace(uint32_t);
::llvm::StringRef stringifyAddressSpace(AddressSpace);
::std::optional<AddressSpace> symbolizeAddressSpace(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAddressSpace() {
  return 8;
}


inline ::llvm::StringRef stringifyEnum(AddressSpace enumValue) {
  return stringifyAddressSpace(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<AddressSpace> symbolizeEnum<AddressSpace>(::llvm::StringRef str) {
  return symbolizeAddressSpace(str);
}
} // namespace pto
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::pto::AddressSpace, ::mlir::pto::AddressSpace> {
  template <typename ParserT>
  static FailureOr<::mlir::pto::AddressSpace> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for PTO Address Space");

    // Symbolize the keyword.
    if (::std::optional<::mlir::pto::AddressSpace> attr = ::mlir::pto::symbolizeEnum<::mlir::pto::AddressSpace>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid PTO Address Space specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::pto::AddressSpace value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::pto::AddressSpace> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::pto::AddressSpace getEmptyKey() {
    return static_cast<::mlir::pto::AddressSpace>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::pto::AddressSpace getTombstoneKey() {
    return static_cast<::mlir::pto::AddressSpace>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::pto::AddressSpace &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::pto::AddressSpace &lhs, const ::mlir::pto::AddressSpace &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace pto {
// Base layout
enum class BLayout : uint32_t {
  RowMajor = 0,
  ColMajor = 1,
};

::std::optional<BLayout> symbolizeBLayout(uint32_t);
::llvm::StringRef stringifyBLayout(BLayout);
::std::optional<BLayout> symbolizeBLayout(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForBLayout() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(BLayout enumValue) {
  return stringifyBLayout(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<BLayout> symbolizeEnum<BLayout>(::llvm::StringRef str) {
  return symbolizeBLayout(str);
}
} // namespace pto
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::pto::BLayout, ::mlir::pto::BLayout> {
  template <typename ParserT>
  static FailureOr<::mlir::pto::BLayout> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Base layout");

    // Symbolize the keyword.
    if (::std::optional<::mlir::pto::BLayout> attr = ::mlir::pto::symbolizeEnum<::mlir::pto::BLayout>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Base layout specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::pto::BLayout value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::pto::BLayout> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::pto::BLayout getEmptyKey() {
    return static_cast<::mlir::pto::BLayout>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::pto::BLayout getTombstoneKey() {
    return static_cast<::mlir::pto::BLayout>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::pto::BLayout &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::pto::BLayout &lhs, const ::mlir::pto::BLayout &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace pto {
// PTO compare mode
enum class CmpMode : uint32_t {
  EQ = 0,
  NE = 1,
  LT = 2,
  LE = 3,
  GT = 4,
  GE = 5,
};

::std::optional<CmpMode> symbolizeCmpMode(uint32_t);
::llvm::StringRef stringifyCmpMode(CmpMode);
::std::optional<CmpMode> symbolizeCmpMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCmpMode() {
  return 5;
}


inline ::llvm::StringRef stringifyEnum(CmpMode enumValue) {
  return stringifyCmpMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CmpMode> symbolizeEnum<CmpMode>(::llvm::StringRef str) {
  return symbolizeCmpMode(str);
}
} // namespace pto
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::pto::CmpMode, ::mlir::pto::CmpMode> {
  template <typename ParserT>
  static FailureOr<::mlir::pto::CmpMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for PTO compare mode");

    // Symbolize the keyword.
    if (::std::optional<::mlir::pto::CmpMode> attr = ::mlir::pto::symbolizeEnum<::mlir::pto::CmpMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid PTO compare mode specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::pto::CmpMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::pto::CmpMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::pto::CmpMode getEmptyKey() {
    return static_cast<::mlir::pto::CmpMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::pto::CmpMode getTombstoneKey() {
    return static_cast<::mlir::pto::CmpMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::pto::CmpMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::pto::CmpMode &lhs, const ::mlir::pto::CmpMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace pto {
// Event ID for Synchronization
enum class EVENT : uint32_t {
  EVENT_ID0 = 0,
  EVENT_ID1 = 1,
  EVENT_ID2 = 2,
  EVENT_ID3 = 3,
  EVENT_ID4 = 4,
  EVENT_ID5 = 5,
  EVENT_ID6 = 6,
  EVENT_ID7 = 7,
};

::std::optional<EVENT> symbolizeEVENT(uint32_t);
::llvm::StringRef stringifyEVENT(EVENT);
::std::optional<EVENT> symbolizeEVENT(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForEVENT() {
  return 7;
}


inline ::llvm::StringRef stringifyEnum(EVENT enumValue) {
  return stringifyEVENT(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<EVENT> symbolizeEnum<EVENT>(::llvm::StringRef str) {
  return symbolizeEVENT(str);
}
} // namespace pto
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::pto::EVENT, ::mlir::pto::EVENT> {
  template <typename ParserT>
  static FailureOr<::mlir::pto::EVENT> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Event ID for Synchronization");

    // Symbolize the keyword.
    if (::std::optional<::mlir::pto::EVENT> attr = ::mlir::pto::symbolizeEnum<::mlir::pto::EVENT>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Event ID for Synchronization specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::pto::EVENT value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::pto::EVENT> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::pto::EVENT getEmptyKey() {
    return static_cast<::mlir::pto::EVENT>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::pto::EVENT getTombstoneKey() {
    return static_cast<::mlir::pto::EVENT>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::pto::EVENT &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::pto::EVENT &lhs, const ::mlir::pto::EVENT &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace pto {
// Global tensor layout (row/col/fractal)
enum class Layout : uint32_t {
  ND = 0,
  DN = 1,
  NZ = 2,
};

::std::optional<Layout> symbolizeLayout(uint32_t);
::llvm::StringRef stringifyLayout(Layout);
::std::optional<Layout> symbolizeLayout(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForLayout() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(Layout enumValue) {
  return stringifyLayout(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Layout> symbolizeEnum<Layout>(::llvm::StringRef str) {
  return symbolizeLayout(str);
}
} // namespace pto
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::pto::Layout, ::mlir::pto::Layout> {
  template <typename ParserT>
  static FailureOr<::mlir::pto::Layout> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Global tensor layout (row/col/fractal)");

    // Symbolize the keyword.
    if (::std::optional<::mlir::pto::Layout> attr = ::mlir::pto::symbolizeEnum<::mlir::pto::Layout>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Global tensor layout (row/col/fractal) specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::pto::Layout value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::pto::Layout> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::pto::Layout getEmptyKey() {
    return static_cast<::mlir::pto::Layout>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::pto::Layout getTombstoneKey() {
    return static_cast<::mlir::pto::Layout>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::pto::Layout &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::pto::Layout &lhs, const ::mlir::pto::Layout &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace pto {
// PTO mask pattern
enum class MaskPattern : uint32_t {
  P0101 = 0,
  P0011 = 1,
  P0110 = 2,
  P0001 = 3,
  P1111 = 4,
  P1010 = 5,
};

::std::optional<MaskPattern> symbolizeMaskPattern(uint32_t);
::llvm::StringRef stringifyMaskPattern(MaskPattern);
::std::optional<MaskPattern> symbolizeMaskPattern(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMaskPattern() {
  return 5;
}


inline ::llvm::StringRef stringifyEnum(MaskPattern enumValue) {
  return stringifyMaskPattern(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MaskPattern> symbolizeEnum<MaskPattern>(::llvm::StringRef str) {
  return symbolizeMaskPattern(str);
}
} // namespace pto
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::pto::MaskPattern, ::mlir::pto::MaskPattern> {
  template <typename ParserT>
  static FailureOr<::mlir::pto::MaskPattern> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for PTO mask pattern");

    // Symbolize the keyword.
    if (::std::optional<::mlir::pto::MaskPattern> attr = ::mlir::pto::symbolizeEnum<::mlir::pto::MaskPattern>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid PTO mask pattern specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::pto::MaskPattern value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::pto::MaskPattern> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::pto::MaskPattern getEmptyKey() {
    return static_cast<::mlir::pto::MaskPattern>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::pto::MaskPattern getTombstoneKey() {
    return static_cast<::mlir::pto::MaskPattern>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::pto::MaskPattern &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::pto::MaskPattern &lhs, const ::mlir::pto::MaskPattern &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace pto {
// Mem Plan Mode
enum class MemPlanMode : uint32_t {
  LOCAL_MEM_PLAN = 0,
  GLOBAL_WORKSPACE_PLAN = 1,
};

::std::optional<MemPlanMode> symbolizeMemPlanMode(uint32_t);
::llvm::StringRef stringifyMemPlanMode(MemPlanMode);
::std::optional<MemPlanMode> symbolizeMemPlanMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMemPlanMode() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(MemPlanMode enumValue) {
  return stringifyMemPlanMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MemPlanMode> symbolizeEnum<MemPlanMode>(::llvm::StringRef str) {
  return symbolizeMemPlanMode(str);
}
} // namespace pto
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::pto::MemPlanMode, ::mlir::pto::MemPlanMode> {
  template <typename ParserT>
  static FailureOr<::mlir::pto::MemPlanMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Mem Plan Mode");

    // Symbolize the keyword.
    if (::std::optional<::mlir::pto::MemPlanMode> attr = ::mlir::pto::symbolizeEnum<::mlir::pto::MemPlanMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Mem Plan Mode specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::pto::MemPlanMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::pto::MemPlanMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::pto::MemPlanMode getEmptyKey() {
    return static_cast<::mlir::pto::MemPlanMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::pto::MemPlanMode getTombstoneKey() {
    return static_cast<::mlir::pto::MemPlanMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::pto::MemPlanMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::pto::MemPlanMode &lhs, const ::mlir::pto::MemPlanMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace pto {
// Pad Mode for LoadOp
enum class PadMode : uint32_t {
  PadNull = 0,
  PadFirstElem = 1,
  PadValue = 2,
};

::std::optional<PadMode> symbolizePadMode(uint32_t);
::llvm::StringRef stringifyPadMode(PadMode);
::std::optional<PadMode> symbolizePadMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForPadMode() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(PadMode enumValue) {
  return stringifyPadMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<PadMode> symbolizeEnum<PadMode>(::llvm::StringRef str) {
  return symbolizePadMode(str);
}
} // namespace pto
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::pto::PadMode, ::mlir::pto::PadMode> {
  template <typename ParserT>
  static FailureOr<::mlir::pto::PadMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Pad Mode for LoadOp");

    // Symbolize the keyword.
    if (::std::optional<::mlir::pto::PadMode> attr = ::mlir::pto::symbolizeEnum<::mlir::pto::PadMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Pad Mode for LoadOp specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::pto::PadMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::pto::PadMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::pto::PadMode getEmptyKey() {
    return static_cast<::mlir::pto::PadMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::pto::PadMode getTombstoneKey() {
    return static_cast<::mlir::pto::PadMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::pto::PadMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::pto::PadMode &lhs, const ::mlir::pto::PadMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace pto {
// Pad value policy
enum class PadValue : uint32_t {
  Null = 0,
  Zero = 1,
  Max = 2,
  Min = 3,
};

::std::optional<PadValue> symbolizePadValue(uint32_t);
::llvm::StringRef stringifyPadValue(PadValue);
::std::optional<PadValue> symbolizePadValue(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForPadValue() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(PadValue enumValue) {
  return stringifyPadValue(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<PadValue> symbolizeEnum<PadValue>(::llvm::StringRef str) {
  return symbolizePadValue(str);
}
} // namespace pto
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::pto::PadValue, ::mlir::pto::PadValue> {
  template <typename ParserT>
  static FailureOr<::mlir::pto::PadValue> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Pad value policy");

    // Symbolize the keyword.
    if (::std::optional<::mlir::pto::PadValue> attr = ::mlir::pto::symbolizeEnum<::mlir::pto::PadValue>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Pad value policy specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::pto::PadValue value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::pto::PadValue> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::pto::PadValue getEmptyKey() {
    return static_cast<::mlir::pto::PadValue>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::pto::PadValue getTombstoneKey() {
    return static_cast<::mlir::pto::PadValue>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::pto::PadValue &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::pto::PadValue &lhs, const ::mlir::pto::PadValue &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace pto {
// PTO Op Pipe
enum class PIPE : uint32_t {
  PIPE_S = 0,
  PIPE_V = 1,
  PIPE_M = 2,
  PIPE_MTE1 = 3,
  PIPE_MTE2 = 4,
  PIPE_MTE3 = 5,
  PIPE_ALL = 6,
  PIPE_MTE4 = 7,
  PIPE_MTE5 = 8,
  PIPE_V2 = 9,
  PIPE_FIX = 10,
  VIRTUAL_PIPE_MTE2_L1A = 11,
  VIRTUAL_PIPE_MTE2_L1B = 12,
  PIPE_NUM = 13,
  PIPE_UNASSIGNED = 99,
};

::std::optional<PIPE> symbolizePIPE(uint32_t);
::llvm::StringRef stringifyPIPE(PIPE);
::std::optional<PIPE> symbolizePIPE(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForPIPE() {
  return 99;
}


inline ::llvm::StringRef stringifyEnum(PIPE enumValue) {
  return stringifyPIPE(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<PIPE> symbolizeEnum<PIPE>(::llvm::StringRef str) {
  return symbolizePIPE(str);
}
} // namespace pto
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::pto::PIPE, ::mlir::pto::PIPE> {
  template <typename ParserT>
  static FailureOr<::mlir::pto::PIPE> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for PTO Op Pipe");

    // Symbolize the keyword.
    if (::std::optional<::mlir::pto::PIPE> attr = ::mlir::pto::symbolizeEnum<::mlir::pto::PIPE>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid PTO Op Pipe specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::pto::PIPE value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::pto::PIPE> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::pto::PIPE getEmptyKey() {
    return static_cast<::mlir::pto::PIPE>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::pto::PIPE getTombstoneKey() {
    return static_cast<::mlir::pto::PIPE>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::pto::PIPE &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::pto::PIPE &lhs, const ::mlir::pto::PIPE &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace pto {
// PTO rounding mode
enum class RoundMode : uint32_t {
  NONE = 0,
  RINT = 1,
  ROUND = 2,
  FLOOR = 3,
  CEIL = 4,
  TRUNC = 5,
  ODD = 6,
  CAST_RINT = 7,
};

::std::optional<RoundMode> symbolizeRoundMode(uint32_t);
::llvm::StringRef stringifyRoundMode(RoundMode);
::std::optional<RoundMode> symbolizeRoundMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForRoundMode() {
  return 7;
}


inline ::llvm::StringRef stringifyEnum(RoundMode enumValue) {
  return stringifyRoundMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<RoundMode> symbolizeEnum<RoundMode>(::llvm::StringRef str) {
  return symbolizeRoundMode(str);
}
} // namespace pto
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::pto::RoundMode, ::mlir::pto::RoundMode> {
  template <typename ParserT>
  static FailureOr<::mlir::pto::RoundMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for PTO rounding mode");

    // Symbolize the keyword.
    if (::std::optional<::mlir::pto::RoundMode> attr = ::mlir::pto::symbolizeEnum<::mlir::pto::RoundMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid PTO rounding mode specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::pto::RoundMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::pto::RoundMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::pto::RoundMode getEmptyKey() {
    return static_cast<::mlir::pto::RoundMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::pto::RoundMode getTombstoneKey() {
    return static_cast<::mlir::pto::RoundMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::pto::RoundMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::pto::RoundMode &lhs, const ::mlir::pto::RoundMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace pto {
// Secondary layout
enum class SLayout : uint32_t {
  NoneBox = 0,
  RowMajor = 1,
  ColMajor = 2,
};

::std::optional<SLayout> symbolizeSLayout(uint32_t);
::llvm::StringRef stringifySLayout(SLayout);
::std::optional<SLayout> symbolizeSLayout(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForSLayout() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(SLayout enumValue) {
  return stringifySLayout(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<SLayout> symbolizeEnum<SLayout>(::llvm::StringRef str) {
  return symbolizeSLayout(str);
}
} // namespace pto
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::pto::SLayout, ::mlir::pto::SLayout> {
  template <typename ParserT>
  static FailureOr<::mlir::pto::SLayout> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Secondary layout");

    // Symbolize the keyword.
    if (::std::optional<::mlir::pto::SLayout> attr = ::mlir::pto::symbolizeEnum<::mlir::pto::SLayout>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid Secondary layout specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::pto::SLayout value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::pto::SLayout> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::pto::SLayout getEmptyKey() {
    return static_cast<::mlir::pto::SLayout>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::pto::SLayout getTombstoneKey() {
    return static_cast<::mlir::pto::SLayout>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::pto::SLayout &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::pto::SLayout &lhs, const ::mlir::pto::SLayout &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace pto {
// PTO High Level Sync Op Type
enum class SyncOpType : uint32_t {
  TLOAD = 0,
  TSTORE_ACC = 1,
  TSTORE_VEC = 2,
  TMOV_M2L = 3,
  TMOV_M2S = 4,
  TMOV_M2B = 5,
  TMOV_M2V = 6,
  TMOV_V2M = 7,
  TMATMUL = 8,
  TVEC = 9,
  TVECWAIT_EVENT = 10,
};

::std::optional<SyncOpType> symbolizeSyncOpType(uint32_t);
::llvm::StringRef stringifySyncOpType(SyncOpType);
::std::optional<SyncOpType> symbolizeSyncOpType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForSyncOpType() {
  return 10;
}


inline ::llvm::StringRef stringifyEnum(SyncOpType enumValue) {
  return stringifySyncOpType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<SyncOpType> symbolizeEnum<SyncOpType>(::llvm::StringRef str) {
  return symbolizeSyncOpType(str);
}
} // namespace pto
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::pto::SyncOpType, ::mlir::pto::SyncOpType> {
  template <typename ParserT>
  static FailureOr<::mlir::pto::SyncOpType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for PTO High Level Sync Op Type");

    // Symbolize the keyword.
    if (::std::optional<::mlir::pto::SyncOpType> attr = ::mlir::pto::symbolizeEnum<::mlir::pto::SyncOpType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid PTO High Level Sync Op Type specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::pto::SyncOpType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::pto::SyncOpType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::pto::SyncOpType getEmptyKey() {
    return static_cast<::mlir::pto::SyncOpType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::pto::SyncOpType getTombstoneKey() {
    return static_cast<::mlir::pto::SyncOpType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::pto::SyncOpType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::pto::SyncOpType &lhs, const ::mlir::pto::SyncOpType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace pto {
// PTO Function Core Type
enum class TFuncCoreType : uint32_t {
  AIC = 1,
  AIV = 2,
  MIX = 3,
  AIC_OR_AIV = 4,
};

::std::optional<TFuncCoreType> symbolizeTFuncCoreType(uint32_t);
::llvm::StringRef stringifyTFuncCoreType(TFuncCoreType);
::std::optional<TFuncCoreType> symbolizeTFuncCoreType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTFuncCoreType() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(TFuncCoreType enumValue) {
  return stringifyTFuncCoreType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TFuncCoreType> symbolizeEnum<TFuncCoreType>(::llvm::StringRef str) {
  return symbolizeTFuncCoreType(str);
}
} // namespace pto
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::pto::TFuncCoreType, ::mlir::pto::TFuncCoreType> {
  template <typename ParserT>
  static FailureOr<::mlir::pto::TFuncCoreType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for PTO Function Core Type");

    // Symbolize the keyword.
    if (::std::optional<::mlir::pto::TFuncCoreType> attr = ::mlir::pto::symbolizeEnum<::mlir::pto::TFuncCoreType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid PTO Function Core Type specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::pto::TFuncCoreType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::pto::TFuncCoreType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::pto::TFuncCoreType getEmptyKey() {
    return static_cast<::mlir::pto::TFuncCoreType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::pto::TFuncCoreType getTombstoneKey() {
    return static_cast<::mlir::pto::TFuncCoreType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::pto::TFuncCoreType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::pto::TFuncCoreType &lhs, const ::mlir::pto::TFuncCoreType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir {
namespace pto {
// PTO Module Core Type
enum class TModuleCoreType : uint32_t {
  AIC = 1,
  AIV = 2,
  MIX = 3,
};

::std::optional<TModuleCoreType> symbolizeTModuleCoreType(uint32_t);
::llvm::StringRef stringifyTModuleCoreType(TModuleCoreType);
::std::optional<TModuleCoreType> symbolizeTModuleCoreType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTModuleCoreType() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(TModuleCoreType enumValue) {
  return stringifyTModuleCoreType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TModuleCoreType> symbolizeEnum<TModuleCoreType>(::llvm::StringRef str) {
  return symbolizeTModuleCoreType(str);
}
} // namespace pto
} // namespace mlir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::pto::TModuleCoreType, ::mlir::pto::TModuleCoreType> {
  template <typename ParserT>
  static FailureOr<::mlir::pto::TModuleCoreType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for PTO Module Core Type");

    // Symbolize the keyword.
    if (::std::optional<::mlir::pto::TModuleCoreType> attr = ::mlir::pto::symbolizeEnum<::mlir::pto::TModuleCoreType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "invalid PTO Module Core Type specification: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::pto::TModuleCoreType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::pto::TModuleCoreType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::pto::TModuleCoreType getEmptyKey() {
    return static_cast<::mlir::pto::TModuleCoreType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::pto::TModuleCoreType getTombstoneKey() {
    return static_cast<::mlir::pto::TModuleCoreType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::pto::TModuleCoreType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::pto::TModuleCoreType &lhs, const ::mlir::pto::TModuleCoreType &rhs) {
    return lhs == rhs;
  }
};
}


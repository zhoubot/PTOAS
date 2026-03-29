from mlir.ir import Context, InsertionPoint, Location, Module
from mlir.dialects import arith, func, memref, pto
from mlir.ir import IndexType, IntegerType, MemRefType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()

            i32 = IntegerType.get_signless(32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)

            vec_memref = MemRefType.get([1, 16], i32, None, vec)
            bound_tile_ty = MemRefType.get([1, 16], i32, None, vec)

            fn_ty = func.FunctionType.get([], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("tile_getval_i32_memref", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c16 = arith.ConstantOp(IndexType.get(ctx), 16).result

                buf = memref.AllocOp(vec_memref, [], [])
                tile = pto.BindTileOp(bound_tile_ty, buf, c1, c16, config=cfg).result

                val = pto.TGetValOp(i32, tile, c0).dst
                pto.TSetValOp(tile, c1, val)

                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())

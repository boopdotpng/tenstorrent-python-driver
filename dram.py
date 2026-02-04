from dataclasses import dataclass
from typing import Callable
from defs import DRAM_ALIGNMENT, DRAM_BARRIER_BASE, Dram, TLBSize
from tlb import TLBConfig, TLBWindow, TLBMode

TILE_R, TILE_C = 32, 32
FACE_R, FACE_C = 16, 16

def _align(x: int, a: int = DRAM_ALIGNMENT) -> int:
    return (x + a - 1) & ~(a - 1)

def _tile_transform(data: bytes, n_tiles: int, forward: bool) -> bytes:
    """Convert between row-major and TILED_NFACES (4 faces per 32x32 tile)."""
    bpe = len(data) // (n_tiles * TILE_R * TILE_C)
    out = bytearray(len(data))
    for t in range(n_tiles):
        toff = t * TILE_R * TILE_C * bpe
        for face_r in range(2):
            for face_c in range(2):
                face_idx = face_r * 2 + face_c
                for r in range(FACE_R):
                    row = face_r * FACE_R + r
                    col = face_c * FACE_C
                    rm_off = toff + (row * TILE_C + col) * bpe
                    tf_off = toff + (face_idx * FACE_R * FACE_C + r * FACE_C) * bpe
                    src_off, dst_off = (rm_off, tf_off) if forward else (tf_off, rm_off)
                    out[dst_off : dst_off + FACE_C * bpe] = data[
                        src_off : src_off + FACE_C * bpe
                    ]
    return bytes(out)

def tilize(data: bytes, n_tiles: int) -> bytes:
    return _tile_transform(data, n_tiles, forward=True)

def untilize(data: bytes, n_tiles: int) -> bytes:
    return _tile_transform(data, n_tiles, forward=False)

@dataclass(frozen=True)
class DramBuffer:
    name: str | None
    addr: int
    size: int
    page_size: int

class DramAllocator:
    def __init__(self, fd: int, dram_tiles: list[tuple[int, int, int]]):
        self.fd = fd
        self.bank_tiles = dram_tiles[:: Dram.TILES_PER_BANK]
        self.next = Dram.DRAM_WRITE_OFFSET
        self.max_page_size = 2 * 1024 * 1024
        self.win = TLBWindow(self.fd, TLBSize.GiB_4)

    def alloc(
        self, size: int, name: str | None = None, *, page_size: int | None = None
    ) -> DramBuffer:
        num_banks = len(self.bank_tiles)
        page_size = min(
            self.max_page_size, page_size or _align((size + num_banks - 1) // num_banks)
        )
        page_size = _align(page_size)
        addr = self.next
        pages = (size + page_size - 1) // page_size
        pages_per_bank = (pages + num_banks - 1) // num_banks
        self.next = _align(self.next + pages_per_bank * page_size)
        return DramBuffer(name=name, addr=addr, size=size, page_size=page_size)

    def alloc_write(
        self, data: bytes, name: str | None = None, *, page_size: int | None = None
    ) -> DramBuffer:
        buf = self.alloc(len(data), name=name, page_size=page_size)
        self.write(buf, data)
        return buf

    def _for_each_page(
        self, buf: DramBuffer, size: int, mode: TLBMode, fn: Callable[[int, int], None]
    ) -> list[tuple[int, int, int]]:
        num_banks = len(self.bank_tiles)
        pages = (size + buf.page_size - 1) // buf.page_size
        touched = []
        for bank_idx, (bank_id, x, y) in enumerate(self.bank_tiles):
            if bank_idx >= pages:
                break
            touched.append((bank_id, x, y))
            self.win.configure(
                TLBConfig(
                    addr=0, start=(x, y), end=(x, y), noc=0, mcast=False, mode=mode
                )
            )
            local_page = 0
            for page_idx in range(bank_idx, pages, num_banks):
                addr = buf.addr + local_page * buf.page_size
                off = page_idx * buf.page_size
                fn(addr, off)
                local_page += 1
        return touched

    def barrier(self, tiles: list[tuple[int, int, int]]):
        for flag in Dram.BARRIER_FLAGS:
            for _, x, y in tiles:
                self.win.configure(
                    TLBConfig(
                        addr=0,
                        start=(x, y),
                        end=(x, y),
                        noc=0,
                        mcast=False,
                        mode=TLBMode.STRICT,
                    )
                )
                self.win.writei32(DRAM_BARRIER_BASE, flag)
                while self.win.readi32(DRAM_BARRIER_BASE) != flag:
                    pass

    def write(self, buf: DramBuffer, data: bytes):
        assert len(data) <= buf.size
        assert (
            buf.page_size >= DRAM_ALIGNMENT
            and (buf.page_size & (DRAM_ALIGNMENT - 1)) == 0
        )
        view = memoryview(data)

        def do_write(addr: int, off: int):
            page = view[off : off + buf.page_size]
            self.win.wc[addr : addr + len(page)] = page

        touched = self._for_each_page(buf, len(data), TLBMode.POSTED, do_write)
        self.barrier(touched)

    def read(self, buf: DramBuffer) -> bytes:
        result = bytearray(buf.size)

        def do_read(addr: int, off: int):
            remaining = buf.size - off
            n = min(buf.page_size, remaining)
            result[off : off + n] = self.win.wc[addr : addr + n]

        self._for_each_page(buf, buf.size, TLBMode.BULK, do_read)
        return bytes(result)

    def close(self):
        self.win.free()

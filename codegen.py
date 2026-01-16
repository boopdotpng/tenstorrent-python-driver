from helpers import TT_HOME
import subprocess 
from enum import Enum, auto
from dataclasses import dataclass

# you need different linker options to build for each core. 
# need to run the compiler three times
# usually, code in trisc0,1,2 is mirrored, but it might be different in some edge cases
class Processor(Enum):
  NCRISC = auto() # data in 
  BRISC = auto() # data out
  # separate because the build process for these differ
  TRISC0 = auto() # unpack (L1 -> dst)
  TRISC1 = auto() # (dst -> dst)
  TRISC2 = auto() # (dst -> L1)

@dataclass
class CompiledKernel:
  pass

class Compiler:
  # taken from tt-metal/jit_build/build.cpp, used for all kernels
  COMMON = [
    "-std=c++17", "-flto=auto", "-ffast-math", "-fno-exceptions",
    "-MMD", "-fno-use-cta-atexit", "-Wall", "-Werror", "-Wno-unknown-pragmas",
    "-Wno-deprecated-declarations", "-Wno-error=multistatement-macros", "-Wno-error=parenthesis",
    "-Wno-error=unused-but-set-variable", "-Wno-unused-variable", "-Wno-unused-function"
  ]
  LDFLAGS = [
    "-Wl", "-z", "max-page-size=16"
  ]
  def __init__(self):
    assert TT_HOME != "", "need path to tt-metal home to compile kernels, set TT_HOME"
    # path may be fickle, use for now 
    self.compiler = TT_HOME/"runtime"/"sfpi"/"compiler"/"bin"/"riscv-tt-elf-g++"
  
  # brisc and ncrisc cores follow a similar compilation process
  def compile_kernel(self, kern: str, processor: Processor):
    pass

  """

  """
  def compile_trisc(self, kern: str):
    pass
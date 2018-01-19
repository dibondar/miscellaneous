import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from fractions import gcd
from pycuda.compiler import SourceModule
from pycuda.tools import dtype_to_ctype
from types import MethodType, FunctionType
from mako.template import Template


class CFuncEval:
    """
    Evaluation of a complicated function using metaprogramming in PyCUDA
    """
    def __init__(self, **kwargs):

        # save all attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        ##########################################################################################
        #
        # Save CUDA constants
        #
        ##########################################################################################

        # Convert real constants into CUDA code

        self.preamble = ''

        for name, value in kwargs.items():
            if isinstance(value, int):
                self.preamble += "#define {} {}\n".format(name, value)

            elif isinstance(value, float):
                self.preamble += "#define {} cuda_real({:.20})\n".format(name, value)

            elif isinstance(value, complex):
                self.preamble += "#define {} cuda_complex({:.20}, {:20})\n".format(name, value.real, value.imag)

        ##########################################################################################
        #
        #   Define block and grid parameters for CUDA kernel
        #
        ##########################################################################################

        #  Make sure that self.max_thread_block is defined
        # i.e., the maximum number of GPU processes to be used (default 512)
        try:
            self.max_thread_block
        except AttributeError:
            self.max_thread_block = 512

        # If the X grid size is smaller or equal to the max number of CUDA threads
        # then use all self.X_gridDIM processors
        # otherwise number of processor to be used is the greatest common divisor of these two attributes
        size_x = self.grid_size

        nproc = (size_x if size_x <= self.max_thread_block else gcd(size_x, self.max_thread_block))

        # CUDA block and grid for functions that act on the whole wave function
        self.cuda_block_grid = {
            "block":(nproc, 1, 1),
            "grid":(size_x // nproc, 1)
        }

        ##########################################################################################
        #
        # Allocate memory for saving the results
        #
        ##########################################################################################

        self.result = gpuarray.zeros(self.grid_size, np.complex128)

        ##########################################################################################
        #
        # Generate the CUDA code and compile
        #
        ##########################################################################################

        self.cuda_params = kwargs.copy()
        self.cuda_params.update(
            # Types for CUDA compilation
            cuda_complex=dtype_to_ctype(self.result.dtype),
            cuda_real=dtype_to_ctype(self.result.real.dtype),
            preamble=self.preamble,
        )

        code = self.cuda_template.render(**self.cuda_params)
        # print(code)
        self.get_f = SourceModule(code).get_function("F")

        self.print_memory_info()

    @classmethod
    def print_memory_info(cls):
        """
        Print the CUDA memory info
        :return:
        """
        print(
            "\n\n\t\tGPU memory Total %.2f GB\n\t\tGPU memory Free %.2f GB\n" % \
            tuple(np.array(pycuda.driver.mem_get_info()) / 2. ** 30)
        )

    def test_gpu(self):
        # calculate f on GPU
        import time
        t = time.time()
        self.get_f(self.result, **self.cuda_block_grid)
        f_gpu = self.result.get()
        print("Running time: {:.2}".format(time.time() - t))
        return f_gpu

    def test_cpu(self):
        # calculate f on CPU
        import itertools

        g = lambda X, Y, Z: np.exp(1j * X) * np.sin(Y) * np.cos(Z)
        G = lambda X, Y, Z: sum(g(*_) for _ in itertools.permutations((X, Y, Z)))

        return sum(
            G(X, Y, Z) for X in np.arange(self.X_min, self.X_max, self.dX)
                for Y in np.arange(self.Y_min, self.Y_max, self.dY)
                for Z in np.arange(self.Z_min, self.Z_max, self.dZ)
        )

    cuda_template = Template("""    
    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES
    
    typedef ${cuda_complex} cuda_complex;
    typedef ${cuda_real} cuda_real;
    
    ${preamble}
    
    __device__ cuda_complex g(const cuda_real X, const cuda_real Y, const cuda_real Z)
    {
        return exp(cuda_complex(0, X)) * sin(Y) * cos(Z);
    }
    
    // symmetrized g
     __device__ cuda_complex G(const cuda_real X, const cuda_real Y, const cuda_real Z)
    {
        cuda_complex result = 0.;

        <%
        from itertools import permutations
        %>
        %for args in permutations(['X', 'Y', 'Z']):
        result += g(${', '.join(args)});
        %endfor
         
        return result;
    }
    
    __global__ void F(cuda_complex *out)
    {
        cuda_complex result = 0.;

        for(cuda_real Z = Z_min; Z < Z_max; Z += dZ)
            for(cuda_real Y = Y_min; Y < Y_max; Y += dY) 
                for(cuda_real X = X_min; X < X_max; X += dX)
                    result += G(X, Y, Z);
        
        // saving the results into the array       
        const int indx = threadIdx.x + blockDim.x * blockIdx.x;
        out[indx] = result;
    }
    """)

##########################################################################################
#
# Example
#
##########################################################################################

if __name__=='__main__':

    obj = CFuncEval(
        grid_size=10,

        dX=1.,
        X_min=-10.,
        X_max=5,

        dY=1.,
        Y_min=-10,
        Y_max=100,

        dZ=1.,
        Z_min=-100.,
        Z_max=100,
    )

    assert np.allclose(obj.test_gpu() - obj.test_cpu(), 0.)

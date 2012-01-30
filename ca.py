#!/usr/bin/python
# -*- coding: utf-8 -*-


try:
    import numpy as np
    import pycuda.driver as cuda
    from pycuda import gpuarray
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
except:
    cuda = None
    np = None



"""
This program is an extensible Conway's game of life. It allows to define
different type of grid (for example 2D or 3D) and more complex rules. 
Each grid inherits an Abstract grid that implement the method (next()) 
to pass for the next configuration. 
Furthermore, each element can be whatever type. In this
example I designed Grid2DBool that represent the simple Conway's game of life,
but could be possible to develop and easily implement more complex grids and
rules.

@author Filippo Squillace
@date 02/12/2011
"""

from mvc import Observable

class Grid(Observable):
    """
    This class represents the abstract grid that implement 
    the template method to generate the next configuration. The rules are
    definied in the abstract method next() and it's not totally
    implemented in this class because depends on the structure of the matrix
    and the type of elements in the grid. In next() Grid class ensures to
    notify the observer and increases the iteration variable.
    """
    def __init__(self):
        Observable.__init__(self)
        self.iteration = 0
    
    def is_done(self):
        """
        Checks out if the evolution is terminated.
        """
        raise NotImplementedError()

    def add_cell(self, coords):
        raise NotImplementedError()

    def remove_cell(self, coords):
        raise NotImplementedError()

    def toggle_cell(self, coords):
        raise NotImplementedError()

    def clear(self):
        """
        Clears all cells of the grid.
        """
        raise NotImplementedError()

    def slice(self, coords, dim):
        """
        Returns a dict with key,value as coords,state of all the cells that are
        in the area specified by coords and dim.
        """
        raise NotImplementedError()

    def next(self):
        self.iteration = self.iteration + 1
        Observable.notify(self)
    
    def previous(self):
        self.iteration = self.iteration - 1
        Observable.notify(self)

    def apply(pattern, coords):
        """
        Applies a pattern into a grid in the given coords.
        """
        raise NotImplementedError()

    def get_num_cells(self):
        raise NotImplementedError()

    def get_range(self):
        raise NotImplementedError()

    def get_state(self):
        """
        Returns the state of the grid. Typically the matrix is stored like an array in numpy.
        """
        raise NotImplementedError()

    def set_state(self, state):
        """
        Set the state in the grid.
        """
        raise NotImplementedError()


class GridGOL(Grid):
    """
    Represents the classical Conway's game of life with 2D grid 
    and each element can be either True (alive) or Fase (death)
    Params:
    dim - (number of rows, number of columns)

    If dim is not specified the grid is unbounded.
    """
    def __init__(self, dim=None):
        Grid.__init__(self)
        self.live_cells = set()
        self.old_live_cells = set()

        self.dim = dim

    def add_cell(self, x, y):
        if self.dim: # Checks if the coord is out of the range
            if x<0 or x>self.dim[0]-1 or y<0 or y>self.dim[1]-1:
                return

        self.live_cells.add((x,y))
        Grid.notify(self)

    def remove_cell(self, x, y):
        if self.dim: # Checks if the coord is out of the range
            if x<0 or x>self.dim[0]-1 or y<0 and y>self.dim[1]-1:
                return

        self.live_cells.remove((x,y))
        Grid.notify(self)

    def toggle_cell(self, x, y):
        if (x,y) in self.live_cells:
            self.remove_cell(x, y)
        else:
            self.add_cell(x, y)

        Grid.notify(self)

    def clear(self):
        self.live_cells.clear()
        Grid.notify(self)

    def apply(self, pattern, x, y):
        h, w = pattern.shape

        itr = pattern.flat
        offx, offy = itr.coords
        for el in itr:
            if el:
                self.add_cell(x+offx, y+offy)
            offx, offy = itr.coords

        Grid.notify(self)

    def get_state(self):
        """
        Return a tuple with first element is the number of iterations and the
        second is the set of live cells.
        """
        return (self.iteration, self.live_cells)

    def set_state(self, state):
        self.iteration, self.live_cells = state
        Grid.notify(self)
    
    def get_num_cells(self):
        return len(self.live_cells)

    def get_range(self):
        return 1 # because the range of the values is True-False

    def slice(self, ipos, dim):
        """
        Returns a dict with key,value as coords,state of all the cells that are
        in the area specified by coords and dim.
        """
        view = {}
        for coords in self.live_cells:
            if coords[0]>=ipos[0] and coords[0]<ipos[0]+dim[0] and\
                    coords[1]>=ipos[1] and coords[1]<ipos[1]+dim[1]:
                rel_coords = (coords[0]-ipos[0], coords[1]-ipos[1])
                view[rel_coords] = True

        return view

    def next(self):
        # Copy the set
        self.old_live_cells = set(self.live_cells)
        self.live_cells = set()

        # Step 1: checks out the next state only for the live cells
        for coords in self.old_live_cells:
            if self.__next_state(self.old_live_cells, coords, True):
                self.live_cells.add(coords)

        # Step 2: checks out the next state for the neighbors of the live cells

        death_neigs_cells = set()
        for coords in self.old_live_cells:
            neigs_cells = self.__get_neighbors(coords)
            death_neigs_cells.update(neigs_cells.difference(self.old_live_cells))

        for coords in death_neigs_cells:
            if self.__next_state(self.old_live_cells, coords, False):
                self.live_cells.add(coords)

        Grid.next(self)

    def previous(self):
        raise NotImplementedError()

    def __get_neighbors(self, coords):
        moore = [(0,-1),(0,1),(-1,-1),(-1,0),(-1,1),(1,-1),(1,0),(1,1)]
        if self.dim:
            neigs = set([ ( (m[0]+coords[0])%self.dim[0] (m[1]+coords[1])%self.dim[1]) for m in moore])
        else:
            neigs = set([ (m[0]+coords[0], m[1]+coords[1]) for m in moore])

        return neigs 

    def __next_state(self, live_cells, coords, el):
        # Gets all information from the neighbors
        neigs_cells = self.__get_neighbors(coords)
        neighbors = len(neigs_cells.intersection(live_cells))

        if el: # el alives
            if neighbors==2 or neighbors==3:
                return True
            if neighbors<2 or neighbors>3:
                return False
        else: # el death
            if neighbors==3:
                return True
    
    def is_done(self):
        return len(self.live_cells)==0 # there is no live cells



class GridCudaGOL(Grid):
    """
    GPU implementation of the classical Conway's game of life with 2D grid 
    and each element can be either True (alive) or Fase (death)
    Params:
    n - number of rows
    m - number of columns
    """
    def __init__(self, dim=None):
        Grid.__init__(self)

        if not cuda or not np:
            raise Exception("Error you can't use CUDA version."+\
            " You need to install numpy and pycuda.")
            return

        if not dim:
            raise Exception("The CUDA implmentation requires a bounded grid.")

        self.matrix_gpu = gpuarray.zeros(dim, np.bool)
        self.old_matrix_gpu = gpuarray.zeros(dim, np.bool)

        self.dim = dim

        source =\
        """
        __global__ void add(bool *a, int x, int y, int m) {
            a[x*m+y] = true;
        }

        __global__ void remove(bool *a, int x, int y, int m) {
            a[x*m+y] = false;
        }
        
        __global__ void toggle(bool *a, int x, int y, int m) {
            a[x*m+y] = not a[x*m+y];
        }

        __device__ int get_num_neighbors(int x, int y, bool* a, int n, int m){
            int neigs = 0;

            int up = (x-1)%n;
            int down = (x+1)%n;

            int left = (y-1)%m;
            int right = (y+1)%m;
            
            if(a[up*m+left])
                neigs++;
            if(a[up*m+y])
                neigs++;
            if(a[up*m+right])
                neigs++;
            
            if(a[x*m+left])
                neigs++;
            if(a[x*m+right])
                neigs++;
            
            if(a[down*m+left])
                neigs++;
            if(a[down*m+y])
                neigs++;
            if(a[down*m+right])
                neigs++;

            return neigs;
        }

        __global__ void next(bool * in, bool* out, int n, int m) {
            // Give an array of cells to analyze it returns the new state of
            // these ones.
            
            const int x = blockDim.x*blockIdx.x + threadIdx.x;
            const int y = blockDim.y*blockIdx.y + threadIdx.y;
            
            if(x>=n || y>=m)
                return;

            const int neigs = get_num_neighbors(x, y, in, n, m);
            if(in[x*m+y]==true){ // cell alives

                if(neigs<2 || neigs>3)
                    out[x*m+y] = false;
            }
            else{ // el death
                if(neigs==3)
                    out[x*m+y] = true;
            }

        }
        """
        mod = SourceModule(source)

        self.add_func = mod.get_function("add")

        self.remove_func = mod.get_function("remove")

        self.toggle_func = mod.get_function("toggle")

        self.next_func = mod.get_function("next")


    def add_cell(self, x, y):
        if self.dim: # Checks if the coord is out of the range
            if x<0 or x>self.dim[0]-1 or y<0 or y>self.dim[1]-1:
                return

        self.add_func(self.matrix_gpu,\
                np.int32(x), np.int32(y),\
                np.int32(self.dim[1]),\
                block=(1,1,1), grid=(1,1))

        Grid.notify(self)

    def remove_cell(self, x, y):
        if self.dim: # Checks if the coord is out of the range
            if x<0 or x>self.dim[0]-1 or y<0 or y>self.dim[1]-1:
                return

        self.remove_func(self.matrix_gpu,\
                np.int32(x), np.int32(y),\
                np.int32(self.m),\
                block=(1,1,1), grid=(1,1))

        Grid.notify(self)

    def toggle_cell(self, x, y):
        if self.dim: # Checks if the coord is out of the range
            if x<0 or x>self.dim[0]-1 or y<0 or y>self.dim[1]-1:
                return

        self.toggle_func(self.matrix_gpu,\
                np.int32(x), np.int32(y),\
                np.int32(self.dim[1]),\
                block=(1,1,1), grid=(1,1))
        
        Grid.notify(self)

    def clear(self):
        self.matrix_gpu.fill(False)
        Grid.notify(self)

    def apply(self, pattern, x,y):
        tmp = np.array([False for _ in range(self.dim[0]*self.dim[1])],\
                dtype=bool).reshape(self.dim[0], self.dim[1])
        h, w = pattern.shape
        tmp[x:x+h, y:y+w] = pattern

        pattern_gpu = gpuarray.to_gpu(tmp)
        self.matrix_gpu = self.matrix_gpu + pattern_gpu

        # Free pattern_gpu
        pattern_gpu.gpudata.free()

        Grid.notify(self)

    def get_state(self):
        """
        Returns a tuple with first element is the number of iterations and the
        second is the matrix array numpy. 
        """
        return (self.iteration ,self.matrix_gpu.get())

    def set_state(self, state):
        it, mat = state
        self.matrix_gpu = gpuarray.to_gpu(mat)
        self.iteration = it
        Grid.notify(self)

    def get_num_cells(self):
        X,Y = np.where(self.matrix_gpu.get()==True)
        return len(X)

    def get_range(self):
        return 1 # because the range of the values is True-False

    def previous(self):
        raise NotImplementedError()

    def slice(self, ipos, dim):
        """
        Returns a dict with key,value as coords,state of all the cells that are
        in the area specified by coords and dim.
        """

        X,Y = np.where(self.matrix_gpu.get()[ipos[0]:ipos[0]+dim[0]+1,\
                ipos[1]:ipos[1]+dim[1]+1]==True)
        view = {}
        for x,y in zip(X,Y):
            view[(x,y)] = True

        return view


    def next(self):
        # Pads out with more threads to ensure all elements of the matrix
        # are assigned to a thread.
        col_blocks = (self.dim[1] + 16-1)/16
        row_blocks = (self.dim[0] + 16-1)/16
        

        # Let's make a copy of the matrix directly on the device memory
        cuda.memcpy_dtod(self.old_matrix_gpu.gpudata, self.matrix_gpu.gpudata,\
                self.matrix_gpu.nbytes) #mem_size)

        self.next_func(self.old_matrix_gpu, self.matrix_gpu,\
                np.int32(self.dim[0]), np.int32(self.dim[1]),\
                block=(16,16,1), grid=(row_blocks,col_blocks))
        
        # Applies notification to the observer
        Grid.next(self)


    def is_done(self):
        return not True #gpuarray.max(self.matrix_gpu) # there is no True













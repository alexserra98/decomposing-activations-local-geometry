from typing import List
class Solution:
    
    def neighbour_indices(self, cell: tuple[int,int], island)->List[tuple[int,int]]:
        neighbors = []
        if cell[0]+1<self.h:
            if self.grid[cell[0]+1][cell[1]] == '1' and (cell[0]+1, cell[1]):# not in island:
                neighbors.append((cell[0]+1, cell[1]))
        
        if 0<=cell[0]-1<self.h:
            if self.grid[cell[0]-1][cell[1]] == '1' and (cell[0]-1, cell[1]): #not in island:
                neighbors.append((cell[0]-1, cell[1]))
        
        if cell[1]+1<self.w:
            if self.grid[cell[0]][cell[1]+1] == '1' and (cell[0], cell[1]+1):# not in island:
                neighbors.append((cell[0], cell[1]+1))
        
        if 0<=cell[1]-1<self.w:
            if self.grid[cell[0]][cell[1]-1] == '1' and (cell[0], cell[1]-1):# not in island:
                neighbors.append((cell[0], cell[1]-1))
        
        return neighbors
            
        
    def find_island(self, land: tuple[int,int])-> List[tuple[int,int]]:
        queue = []
        island = []
        queue.append(land)
        while len(queue) > 0:
            head = queue[0]
            queue.extend(self.neighbour_indices(head, island))
            x = queue.pop(0)
            island.append(x)
   
        
        return  island      

    def delete_island(self, island: List[tuple[int,int]])-> List[List[str]]:
        for row in range(self.h):
            for column in range(self.w):
                if (row,column) in island:
                    self.grid[row][column] = '0'
        return self.grid        

    def search_land(self) -> tuple[int,int] | bool:
        for row in range(self.h):
            for column in range(self.w):
                if self.grid[row][column] == '1':
                    return (row, column)
        return False
    
    def numIslands(self, grid: List[List[str]]) -> int:
        self.h, self.w = len(grid), len(grid[0])
        self.grid = grid
        num_islands = 0
        while land := self.search_land():
                island = self.find_island(land)
                num_islands += 1
                self.delete_island(island)
        return num_islands

input = [["1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","0","1","0","1","1"],["0","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","0"],["1","0","1","1","1","0","0","1","1","0","1","1","1","1","1","1","1","1","1","1"],["1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1"],["1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","0","1","0","1","1"],["0","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","0"],["1","0","1","1","1","0","0","1","1","0","1","1","1","1","1","1","1","1","1","1"],["1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1"],["1","0","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1"]]
solution = Solution()

print(solution.numIslands(input))
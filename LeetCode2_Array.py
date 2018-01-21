# Valid Sudoku
# Determine if a Sudoku is valid, according to: Sudoku Puzzles - The Rules
# http://sudoku.com.au/TheRules.aspx .
# The Sudoku board could be partially filled, where empty cells are filled with the character '.'.
# 填充数独里面的数字，空单元格用 ',' 填充
# 为了便于计算 我直接改成了0
# 5 3 0  0 7 0  0 0 0
# 6 0 0  1 9 5  0 0 0
# 0 9 8  0 0 0  0 6 0
#
# 8 0 0  0 6 0  0 0 3
# 4 0 0  8 0 3  0 0 1
# 7 0 0  0 2 0  0 0 6
#
# 0 6 0  0 0 0  2 8 0
# 0 0 0  4 1 9  0 0 5
# 0 0 0  0 8 0  0 7 9
#
# 切 我还以为是数独的解法
# 其实这个题只是让你判断是否是一个有效的数独
# 意思就是 已有的数据里面 横竖 小方块是否有重复的数据
# 而不需要填充数据
# 用一个长度为9的boolean数组
def valid_sudoku(A=[]):
    # 检查竖
    for i in range(9):
        # 重置判读数组
        B = [False] * 9
        for j in range(9):
            if not check_valid(B, A[i][j]):
                return False
    # 检查横
    for i in range(9):
        # 重置判断数组
        B = [False] * 9
        for j in range(9):
            if not check_valid(B, A[j][i]):
                return False
    # 检查小方块
    for i in range(3):
        for j in range(3):
            B = [False] * 9
            for a in range(3):
                for b in range(3):
                    x = i * 3 + a
                    y = j * 3 + b
                    if not check_valid(B, A[x][y]):
                        return False
    return True


# 检查某个点的有效性
def check_valid(A=[], i=0):
    if i == 0:
        return True
    if i < 1 or i > 9 or A[i - 1]:
        return False
    A[i - 1] = True
    return True


A = [[5, 3, 0, 0, 7, 0, 0, 0, 0], [6, 0, 0, 1, 9, 5, 0, 0, 0], [0, 9, 8, 0, 0, 0, 0, 6, 0],
     [8, 0, 0, 0, 6, 0, 0, 0, 3], [4, 0, 0, 8, 0, 3, 0, 0, 1], [7, 0, 0, 0, 2, 0, 0, 0, 6],
     [0, 6, 0, 0, 0, 0, 2, 8, 0], [0, 0, 0, 4, 1, 9, 0, 0, 5], [0, 0, 0, 0, 8, 0, 0, 7, 9]]


# Trapping Rain Water
# Given n non-negative integers representing an elevation map where the width of each bar is 1, compute
# how much water it is able to trap after raining.
# For example, Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6.
# 雨水收集器
# 网上有图 看着更清楚 (建议自己百度一下)
# 基本思路就是 每根柱子 如果左边或者右边 没有比他高的数  那么就不能存储雨水
# 如果左右两边都有比他高的数  那么 就用左右两边中小的那个 减去他本身 就可以得到本柱子能存储多少雨水了
# 然后再进行叠加即可
#

def trap_rain_water(L=[]):
    result = 0
    for i in range(len(L)):
        left = 0
        right = 0
        x = i
        y = i
        while x > 0:
            x -= 1
            if L[x] > L[i] and L[x] > left:
                left = L[x]
        while y < len(L) - 1:
            y += 1
            # print(y)
            if L[y] > L[i] and L[y] > right:
                right = L[y]
        if left == 0 or right == 0:
            continue
        result += (min(left, right) - L[i])
    print('这个模型 应该能收集的雨水是 %s  ' % result)


# 该题的第二种思路
# 先遍历一遍 找到最高点 （有多个最高点 就取第一个）
# 然后将数组分成两边 分别依次处理 左右两边即可
# 就不写代码了 思路很简单
def trap_rain_water_2(L=[]):
    pass

# Rotate Image
# You are given an n * n 2D matrix representing an image.
# Rotate the image by 90 degrees (clockwise).
# Follow up: Could you do this in-place?
# 有一个 n * n 的2D图片 让这个图片旋转90度
#
print(trap_rain_water([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))

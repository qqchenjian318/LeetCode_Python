# LeetCode里面的链表相关的算法题


# Add Two Numbers
# You are given two linked lists representing two non-negative numbers. The digits are stored in reverse
# order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
# Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
# Output: 7 -> 0 -> 8
# 这个题主要是类似数据在机器中存储的方式，比如常见的342，在链表中是逆向存储的，所以就成了 2—>4—>3这样
# 同样 5—>6—>4 就是465，我们就会发现342 + 465 = 807，在十位上相加是超过10向前进一，但是链表是逆向的，所以就是向后进一
#
# 其实可以将两个链表转成整型 相加 然后将结果倒序 插入新的链表中即可
#
# 链表的节点
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


def two_number_add(a, b):
    numLine1 = []
    numLine2 = []
    while a is not None:
        numLine1.append(a.val)
        a = a.next
    while b is not None:
        numLine2.append(b.val)
        b = b.next

    num1 = 0
    num2 = 0
    for n in range(len(numLine1)):
        num1 += numLine1[n] * (10 ** n)  # 10 ** n表示 10 的n次方

    for m in range(len(numLine2)):
        num2 += numLine2[m] * (10 ** m)

    num3 = num1 + num2
    # 构造链表返回
    L3 = ListNode
    first = ListNode(0)
    L3.val = 'start'
    L3.next = first
    while num3 >= 10:
        temp = ListNode(None)
        first.val = num3 % 10
        first.next = temp
        first = temp
        num3 = num3 // 10

    first.val = num3 % 10  # 此时的num3 < 10
    return L3


a = ListNode(0)
b = ListNode(0)
c = ListNode(0)
a.val = 2
a.next = b
b.val = 4
b.next = c
c.val = 3
c.next = None

l = ListNode(0)
m = ListNode(0)
n = ListNode(0)
l.val = 5
l.next = b
m.val = 6
m.next = c
n.val = 4
n.next = None


# Reverse Linked List II
# Reverse a linked list from position m to n. Do it in-place and in one-pass.
# For example: Given 1->2->3->4->5->nullptr, m = 2 and n = 4,
# return 1->4->3->2->5->nullptr.
# Note: Given m, n satisfy the following condition: 1 <= m <= n <= length of list
#
# 这个题就是reverse一个链表从 值m 到 值n，在原链上一次完成
#
# 思路就是从时候
#
def reverse_link(a, m, n):
    temp2 = None
    new = ListNode(0)
    new.val = 'start'
    new.next = a
    for i in range(m - 1):
        a = a.next
    newLink = a  # newLink 现在是2
    # 从new 开始 就是一个全新的链表
    while m <= n:
        m += 1
        # 此时的a 其实就是第m个节点
        temp = a.next
        a.next = temp2
        temp2 = a
        a = temp
    newLink.next = a
    new.next.next = temp2
    return new


a = ListNode(0)
b = ListNode(0)
c = ListNode(0)
d = ListNode(0)
e = ListNode(0)
f = ListNode(0)

a.val = 1
a.next = b
b.val = 4
b.next = c
c.val = 3
c.next = d
d.val = 2
d.next = e
e.val = 5
e.next = f
f.val = 2
f.next = None


# Partition List
# Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater
# than or equal to x.
# You should preserve the original relative order of the nodes in each of the two partitions.
# For example, Given 1->4->3->2->5->2 and x = 3, return 1->2->2->4->3->5.
#
# 给定一个链表和一个x，分离这个链表，将比x小的节点都放在前面，大于等于的放到后面 每部分元素的原始相对位置不变
# 好吧，思路还是相对简单的 遍历链表大的就放在大的链表那边 小的就放在小的链表那边 然后将两个组合起来即可

def partition_list(a, x):
    small = None  # 头节点
    big = None  # 头节点
    right = None
    first = None
    while a is not None:
        print(a.val)
        if a.val < x:
            if small is not None:
                small.next = a
                small = small.next
            else:
                small = a
                first = a
        else:
            if big is not None:
                big.next = a
                big = big.next
            else:
                big = a
                right = a
        a = a.next
    small.next = right
    big.next = None
    return first


#  Remove Duplicates from Sorted List
#  Given a sorted linked list, delete all duplicates such that each element appear only once.
#  For example,
#  Given 1->1->2, return 1->2.
#  Given 1->1->2->3->3, return 1->2->3.
#
#  将重复的元素从一个有序的链表中移除掉
#  有一个有序的链表，删除其中所有的重复元素
#
def remove_dup_list(a):
    temp = None
    last = None
    first = a
    while a is not None:
        # print(a.val)
        if temp != a.val:
            # 说明是不重复的
            if last is not None:
                last.next = a.next
            last = a

        temp = a.val
        a = a.next
    return first


# Remove Duplicates from Sorted List II
# Given a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers
# from the original list.
# For example,
# Given 1->2->3->3->4->4->5, return 1->2->5.
# Given 1->1->1->2->3, return 2->3.

# 删除链表中的重复元素，和上题类似
# 不同的是 将有重复元素的节点都删除掉
# 思路
# 利用递归解题
# 如果节点 a的值和a的下个节点p的值相同
# 那么 就将p指向p的下一个节点 直到返回不相同的节点
# 如果不等
# 那么 a的下一个节点 等于下一个不相同的节点
# 所以整个函数 其实返回的就是 每一段的不相同的节点

def remove_dup_node_2(a):
    if a is None or a.next is None:
        return a
    p = a.next
    if a.val == p.val:
        # 两者相等
        while p is not None and a.val == p.val:
            p = p.next
        return remove_dup_node_2(p)
    else:
        # 两者不等
        a.next = remove_dup_node_2(p)
        return a


# Rotate List
# Given a list, rotate the list to the right by k places, where k is non-negative.
# For example: Given 1->2->3->4->5->nullptr and k = 2, return 4->5->1->2->3->nullptr.
#
# 旋转list
# 从第k个节点开始旋转list
#
# 思路：首先 将链表形成一个闭环
# 然后算出 需要第几个那里断开
# 然后从断开处断开 将下一个节点作为首节点 即可

def rotate_list(a, k):
    temple = a
    count = 0
    while temple.next is not None:
        count += 1
        temple = temple.next
    k = count - k % count
    temple.next = a
    for i in range(k):
        temple = temple.next
    a = temple.next
    temple.next = None

    return a

# Remove Nth Node From End of List
# Given a linked list, remove the nth node from the end of list and return its head.
# For example, Given linked list: 1->2->3->4->5, and n = 2.
# After removing the second node from the end, the linked list becomes 1->2->3->5.
# Note:
# Given n will always be valid.
# Try to do this in one pass.

# 移除掉倒数 第n个node
# 因为是移除倒数第n个节点
# 所以可以两个指针  第一个p指针先移动n步  然后两个一起移动
# 然后将q指针的next指向 下下个即可

# 下面算法没有考虑 节点总数小于n的情况
# 还有n《= 0 的情况


def remove_n_node(a, n):
    if n <= 0:
        return 'n值小于0了'
    head = a
    p = a
    q = a

    for i in range(n):
        p = p.next
    while p.next is not None:
        p = p.next
        q = q.next
    q.next = q.next.next
    return head


# Swap Nodes in Pairs
# Given a linked list, swap every two adjacent nodes and return its head.
# For example, Given 1->2->3->4, you should return the list as 2->1->4->3.
# Your algorithm should use only constant space. You may not modify the values in the list, only nodes
# itself can be changed.

# 一对对的交换节点
# 有一个链表交互相邻两个节点
# 例如 1 2 3 4  result 2 1 4 3
# 规定是 不能增加空间 你不能修改链表中的值 只能改变节点本身
# 这道题的思路就是 每次跳两个节点
def swap_node(a):
    helper = ListNode(0)
    helper.next = a
    pre, cur = helper, a
    while cur and cur.next:
        pre.next = cur.next
        cur.next = pre.next.next
        pre.next.next = cur
        pre, cur = cur, cur.next
    return helper.next

# Reverse Nodes in k-Group
# Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.
# If the number of nodes is not a multiple of k then left-out nodes in the end should remain as it is.
# You may not alter the values in the nodes, only nodes itself may be changed.
# Only constant memory is allowed.
# For example, Given this linked list: 1->2->3->4->5
# For k = 2, you should return: 2->1->4->3->5
# For k = 3, you should return: 3->2->1->4->5
# 上面题目的衍生题目，反转k长度的链表
# 意思就是 将链表 每k个节点进行位置转换
def swap_k_node(head):
    pass



a = ListNode(0)
b = ListNode(0)
c = ListNode(0)
d = ListNode(0)
e = ListNode(0)
f = ListNode(0)

a.val = 1
a.next = b
b.val = 2
b.next = c
c.val = 3
c.next = d
d.val = 4
d.next = e
e.val = 5
e.next = None

result = swap_node(a)
count = 0
while result is not None and count < 10:
    print(result.val)
    result = result.next
    count += 1



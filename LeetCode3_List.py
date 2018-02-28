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
def reverse_link(a , m, n):
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
b.val = 2
b.next = c
c.val = 3
c.next = d
d.val = 4
d.next = e
e.val = 5
e.next = None
result = reverse_link(a, 2, 5)
while result is not None:
    print(result.val)
    result = result.next
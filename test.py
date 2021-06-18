import torch
import random

def labelTranslate(label1, label2):
    if torch.equal(label1, torch.tensor([[1, 0, 0, 0]])): # F_young
        str1 = 'FY'
    elif torch.equal(label1, torch.tensor([[0, 1, 0, 0]])): # F_middle
        str1 = 'FM'
    elif torch.equal(label1, torch.tensor([[0, 0, 1, 0]])): # M_young
        str1 = 'MY'
    elif torch.equal(label1, torch.tensor([[0, 0, 0, 1]])): # M_middle
        str1 = 'MM'

    if torch.equal(label2, torch.tensor([[1, 0, 0, 0]])): # F_young
        str2 = 'FY'
    elif torch.equal(label2, torch.tensor([[0, 1, 0, 0]])): # F_middle
        str2 = 'FM'
    elif torch.equal(label2, torch.tensor([[0, 0, 1, 0]])): # M_young
        str2 = 'MY'
    elif torch.equal(label2, torch.tensor([[0, 0, 0, 1]])): # M_middle
        str2 = 'MM'
    
    return str1 + '2' + str2

l1 = torch.tensor([[1, 0, 0, 0]])
l2 = torch.tensor([[0, 1, 0, 0]])

print(labelTranslate(l1,l2))
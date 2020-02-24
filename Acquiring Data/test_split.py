
'''
O9I1Yb5
O9Br1Yb5
O9Se1Eu4
F7S2Tm1
O8Se1Yb4
O8Cl1Tm3
'''

a='Ce1.3333HLa2CuO4.4'

a='O4P1.33333333U1'

a='F1.33333333Sn1Pb1.33333333'


print(a)

a=a+'A'
s=0
blist=[]

for ii in range(1,len(a)):

  s1 = str(a[ii])
  num1 = ord(s1)

  #Capital Letter:
  if num1 >= 65:
    if num1 <= 90:
      blist.append(a[s:ii])
      s=ii

print(blist)

clist=[]
dlist=[]

for ii in range(0,len(blist)):
  strtemp = blist[ii]
  #print(strtemp)

  if len(strtemp) == 1:
    clist.append(strtemp)
    dlist.append(1)
  else:

    #The second character is "number":
    if (ord(strtemp[1])) >= 48:
      if (ord(strtemp[1])) <= 57:
        clist.append(strtemp[0])
        #dlist.append(int(strtemp[1:])) 
        dlist.append(float(strtemp[1:])) 

    #The second character is "little letter":
    if(ord(strtemp[1])) >= 97:
      if(ord(strtemp[1])) <= 122:
        clist.append(strtemp[0:2])

        if len(strtemp) == 2:
          dlist.append(1)
        else:
          #dlist.append(int(strtemp[2:]))
          dlist.append(float(strtemp[2:]))
          
print(clist)
print(dlist)

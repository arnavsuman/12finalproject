def bubbleSort(arr):
    n = len(arr)

    for i in range(n-1):
        for j in range(0, n-i-1):

            if arr[j] > arr[j + 1] :
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
a=int(input('enter total no: '))
arr=[]
for i in range(a):
    b=int(input('enter no: '))
    arr.append(b)
arr2=arr
bubbleSort(arr)
print('buuble sort:')
print(arr)
def selectsort(A):
    for i in range(len(A)):
        
        # Find the minimum element in remaining 
        # unsorted array
        min_idx = i
        for j in range(i+1, len(A)):
            if A[min_idx] > A[j]:
                min_idx = j
                
        # Swap the found minimum element with 
        # the first element        
        A[i], A[min_idx] = A[min_idx], A[i]
selectsort(arr2)
print('selection sort:')
print(arr2)
'''


'''

list1=[]
a=int(input('enter length of list: '))
for i in range(a):
    b=int(input('enter no: '))
    list1.append(b)

c=int(input('enter element to delete: '))
print('list before')
print(list1)
def remove(list1, element):
    res = [i for i in list1 if i != element]
    return res
print('list after')
print(remove(list1,c))


list1=[1,1,1,3,5,3,67,86,11,23]
def push(list):
    num=int(input('enter no to enter: '))
    list.append(num)

def pop(list):
    list.pop()

def display(list):
    print(list)

def count(list):
    num=int(input('enter element to count:'))
    count=0
    for i in list:
        if i ==num:
            count=count+1
    print('the element ',str(num),' appeared ', str(count),' times.')
while True:
    print('Type 1 for Push. ')
    print('tyype 2 to pop last element')
    print('type 3 to display all no.')
    print('type 4 to count')
    print('type 5 to exit')
    print()
    sel=int(input('enter choice: '))

    if sel == 1:
        push(list1)
    if sel == 2:
        pop(list1)
    if sel== 3:
        display(list1)
    if sel== 4:
        count(list1)
    if sel== 5:
        break

list1=[]
a=int(input('enter length of list: '))
for i in range(a):
    b=int(input('enter no: '))
    list1.append(b)

num= int(input('enter elememnt to delete: '))
print(list1)
list1.remove(num)
print(list1)

a= input("Enter a word: ")
s=[]
for i in a:
    s.append(i)
list1 = ''
while not len(s)==0:
  list1 += s.pop()

if list1 == a:
    print("The word is a palindrome")
else:
    print("It's not a palindrome")

employee=[]
def push():
    empno=int(input("Enter empno: "))
    name=input("Enter name: ")
    emp=(empno,name)
    employee.append(emp)
    print()
def pop():
    if(employee==[]):
        print("Underflow / Employee Stack in empty")
    else:
        empno,name=employee.pop()
        print("poped element is ")
        print("empno ",empno," name ",name)
    print()
def peek():
    num=int(input('Enter employee no. to search: '))
    if employee==[]:
        print("Empty , No employee to display")
    else:
        for j in employee:
            if j[0]==num:
                print('Employee name is: ',str(j[1]))
                print('Employee no is: ',str(j[0]))
    print()
while True:
    print("1. Push")
    print("2. Pop")
    print("3. Peep")
    print("4. Exit")
    ch=int(input("Enter your choice: "))
    if(ch==1):
        push()
    elif(ch==2):
            pop()
    elif(ch==3):
            peek()
    elif(ch==4):
        print("End")
        break


import mysql.connector 
db = mysql.connector.connect(host='localhost', user='root', password='Arnavcool1', database ='product')
cur=db.cursor()

print('1.')
cur.execute('SELECT * from product_table;')
for i in cur.fetchall():
    print(i)

print('2.')
cur.execute('Select Iname, Price from product_table')
for i in cur.fetchall():
    print(i)

print('3.')
cur.execute("Select * from product_table where Iname='Soap'; ")
print(cur.fetchone())

print('4.')
cur.execute("Select * from product_table where Iname like 's%'; ")
for i in cur.fetchall():
    print(i)

print('5.')
cur.execute("Select Itemno , Iname , ( price * quantity) as 'Total price' from product_table;")
for i in cur.fetchall():
    print(i)

print('6.')
cur.execute("select * from product_table order by Iname ;")
for i in cur.fetchall():
    print(i)

print('7.')
cur.execute("select Iname, Price from product_table order by Price DESC ;")
for i in cur.fetchall():
    print(i)


print('.8')
cur.execute("select Iname from product_table where Price between 50 and 100;")
for i in cur.fetchall():
    print(i)

print('9.')
cur.execute("update product_table set Price = Price*0.95;")
cur.execute('select * from product_table;')
for i in cur.fetchall():
    print(i)

import mysql.connector 
db = mysql.connector.connect(host='localhost', user='root', password='Arnavcool1', database ='product')
cur=db.cursor()

print('1.')
cur.execute('ALTER TABLE product_table ADD totalprice decimal (10,2);')
db.commit()
print('2.')
cur.execute('ALTER TABLE product_table MODIFY COLUMN Iname varchar(25);')
db.commit()

print('3. ')
cur.execute("Select ( price * quantity) as 'Total price' from product_table;")
price22=[]

for i in cur.fetchall():
    price22.append(i)

query="INSERT INTO product_table(totalprice) values(%s);"
cur.executemany(query, price22)
db.commit()
cur.execute("Select totalprice from product_table")
for i in cur.fetchall():
    print(i)
db.commit()

print('4.')
cur.execute('SELECT * FROM product_table WHERE Price = (SELECT MAX(Price) FROM product_table);')
print(cur.fetchone())
print()
db.commit()

print('5.')
cur.execute("DELETE FROM product_table WHERE Iname='Powder';")
for i in cur.fetchall():
    print(i)
db.commit()

print('6.')
cur.execute('ALTER TABLE product_table DROP COLUMN totalprice;')
for i in cur.fetchall():
    print(i)
db.commit()

print('7. ')
cur.execute('DROP TABLE product_table;')

print('8. ')
cur.execute('DROP DATABASE product;')


import mysql.connector 
db = mysql.connector.connect(host='localhost', user='root', password='Arnavcool1', database ='product')
cur=db.cursor()

cur.execute('Select Item.Icode, IName, BNAME from Item, BNAME Where Item.ICode = BNAME.Icode and Item.price between 25000 and 30000;')
for i in cur.fetchall():
    print(i)

cur.execute("Select Item.Icode, price, BNAME from Item,BNAME where Item.ICode = BNAME.Icode and Iten.Iname like 'Television';")
for i in cur.fetchall():
    print(i)

cur.execute("Update item set Price = Price * 1.10 ;")
for i in cur.fetchall():
    print(i)

cur.execute("SELECT DISTINCT BNAME from BRAND;")
for i in cur.fetchall():
    print(i)



import mysql.connector 
db = mysql.connector.connect(host='localhost', user='root', password='Arnavcool1', database ='product')
cur=db.cursor()

data=[(501, 'R Jain', 98, 'M', 111),(545, 'Kavita', 73, 'F',333),(705, 'K.rashika', 85, 'F', 111),(754, 'Rahul Goel', 60, 'M', 444),(892, 'Sahil Jain', 78, 'M',333),(935, 'Rohan Saini', 85, 'M', 222), (955, 'Anjali',64, 'F',444), (983, 'Sneha Aggarwal',80,'F',222)]

query="INSERT INTO students( Adno, Name , Average, Gender, Scode ) values(%s,%s,%s,%s,%s); "
cur.executemany(query, data)
db.commit()

import mysql.connector 
db = mysql.connector.connect(host='localhost', user='root', password='Arnavcool1', database ='product')
cur=db.cursor()
#1.
cur.execute("SELECT *  from students;")
for i in cur.fetchall():
    print(i)
#2.
cur.execute("SELECT *  from students where Name='Rohan Saini';")
for i in cur.fetchall():
    print(i)
#3.
cur.execute("SELECT count(distinct Adno) from students;")
cur.fetchone()
print('Total no. of students are: ', str(cur.fetchone()))
#4.
cur.execute("SELECT count(distinct Adno) from students where Gender='M';")
print('Total no. of Male students are: ', str(cur.fetchone()))

cur.execute("SELECT count(distinct Adno) from students where Gender='F';")
print('Total no. of Female students are: ', str(cur.fetchone()))
#5.
cur.execute("select * from students order by Name ;")
for i in cur.fetchall():
    print(i)
#6.
cur.execute("select * from students order by Average DESC ;")
for i in cur.fetchall():
    print(i)
#7.
cur.execute("select * from students where Name like 'K%' ;")
for i in cur.fetchall():
    print(i)
#8.
cur.execute("select * from students where Name like '%I' ;")
for i in cur.fetchall():
    print(i)
#9.
cur.execute("select Adno, Name, (Average*5) as 'Total_Marks' from students;")
for i in cur.fetchall():
    print(i)
#10.
cur.execute("select * from students where Average between 80 and 90;")
for i in cur.fetchall():
    print(i)


import mysql.connector 
db = mysql.connector.connect(host='localhost', user='root', password='Arnavcool1', database ='product')
cur=db.cursor()
#1.
cur.execute("SELECT *  from students where Average>80 and Scode< 333;")
for i in cur.fetchall():
    print(i)
#2.
cur.execute("SELECT Name, Average  from students where Scode between 222 and 333;")
for i in cur.fetchall():
    print(i)
#3.
cur.execute("SELECT SUM(Average) from students;")
for i in cur.fetchone():
    print(i)
#4.
cur.execute("SELECT MAX(Average) from students;")
for i in cur.fetchone():
    print(i)
#5.
cur.execute("SELECT MIN(Average) from students;")
for i in cur.fetchone():
    print(i)
#6.
cur.execute("SELECT AVG(Average) from students where Gender='F';")
print('Average marks of Female students are: ', str(cur.fetchone()))
cur.execute("SELECT AVG(Average) from students where Gender='M';")
print('Average marks of male students are: ', str(cur.fetchone()))
#7.
cur.execute("SELECT min(distinct scode), max(distinct scode), sum(distinct scode) from students;")
for i in cur.fetchall():
    print(i)
#8.
cur.execute("SELECT count(distinct scode) from students;")
for i in cur.fetchone():
    print(i)

import mysql.connector 
db = mysql.connector.connect(host='localhost', user='root', password='Arnavcool1', database ='product')
cur=db.cursor()

#1.
cur.execute("SELECT BookName, AuthorName from BOOKS where Publisher='ABC';")
for i in cur.fetchall():
    print(i)

#2.
cur.execute("select * from BOOKS order by Price DESC ;")
for i in cur.fetchall():
    print(i)

#3.
cur.execute("SELECT BOOKS.Book_ID, BOOKS.BookName, BOOKS.AuthorName, BOOKS.Publisher, BOOKS.Price, BOOKS.QTY, ISSUES.Oty_Issued FROM BOOKS LEFT JOIN ISSUES ON BOOKS.Book_ID=ISSUES.Book_ID;")
for i in cur.fetchall():
    print(i)

#4.
cur.execute("select distinct AuthorName ,MIN(Price) from BOOKS order by Price DESC ;")
for i in cur.fetchall():
    print(i)

#5.
cur.execute("SELECT  BOOKS.Price, ISSUES.Oty_Issued FROM BOOKS LEFT JOIN ISSUES ON BOOKS.Book_ID=ISSUES.Book_ID where Oty_Issued=5;")
for i in cur.fetchall():
    print(i)



import mysql.connector 
db = mysql.connector.connect(host='localhost', user='root', password='Arnavcool1')
cur=db.cursor()

cur.execute('CREATE DATABASE library;')
cur.execute('CREATE TABLE book (Book_id int(100) primary key, book_title varchar(255), Author varchar(255), Price int(25), Qty int(25));')

def update():
    id=int(input('enter Book_id: '))
    title=input('enter book_title: ')
    author=input('enter Author: ')
    price=int(input('enter price: '))
    qty=int(input('enter qty: '))
    cur.excute("UPDATE book SET book_title= (%s), Author=(%s), Price=(%s), Qty=(%s) WHERE CustomerID = (%s)").values(title, author, price, qty, id);
    db.commit()
def add():
    id=int(input('enter Book_id: '))
    title=input('enter book_title: ')
    author=input('enter Author: ')
    price=int(input('enter price: '))
    qty=int(input('enter qty: '))
    rec=(id, title, author, price, qty)
    query="INSERT INTO book values(%s,%s,%s,%s,%s);"
    cur.execute(query, rec)
    db.commit()
def search():
    id=int(input('enter Book_id: '))
    cur.execute('select * from book where Book_id=(%s)').value(id)
def delete():
    id=int(input('enter id: '))
    cur.execute('delete from book where Book_id=(%s)').value(id)
    db.commit()
def display():
    cur.execute("select * from book;")
    for i in cur.fetchall():
        print(i)
    print()

print()
print('enter 1. add a record based on cust_id.')
print('enter 2. search a record.')
print('enter 3. update a record.')
print('enter 4 to Delete record based on book_Id.')
print('enter 5 to Display all the record.')
print('enter 6 to exit.')
print()
while True:
    ch=int(input('enter your choice: '))
    if ch==1:
        add()
    elif ch==2:
        search()
    elif ch==3:
        update()
    elif ch==4:
        delete()
    elif ch==5:
        display()
    elif ch==6:
        break









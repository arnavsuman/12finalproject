'''
from ast import Pass
import mysql.connector 
from tkinter import messagebox
db = mysql.connector.connect(host='localhost', user='root', password='Arnavcool1', database='crypto')

cur=db.cursor()
#

acc_id_t='test2'#delte
how_cc_t=0.2#delete
coin='BTC'#delete
history_str="BUY: 1.0BTC, BUY: 2.9ETH, SELL: 0.5BTC, BUY: 0.5BTC, BUY: 12.0DOGE, SELL: 0.5BTC, BUY: 0.5BTC, SELL: 0.99BTC, BUY: 0.99BTC, SELL: 0.5BTC, SELL: 0.1BTC, BUY: 0.6BTC, SELL: 0.1BTC, BUY: 0.1BTC, BUY: 0.018BTC, SELL: 0.02BTC, SELL: 0.2BTC, BUY: 0.2BTC, SELL: 0.1BTC, BUY: 0.1BTC, SELL: 0.5BTC, BUY: 0.5BTC, BUY: 100.0XRP, SELL: 0.2BTC, BUY: 0.21BTC, BUY: 0.21BTC, BUY: 0.01BTC, BUY: 0.01BTC, BUY: 0.0013BTC, SELL: 0.02BTC, SELL: 0.02BTC, BUY: 0.02BTC, SELL: 1.0BTC, BUY: 1.0BTC, SELL: 1.0BTC, SELL: 1.0BTC, BUY: 1.0BTC, BUY: 1.0SOL, SELL: 0.02BTC, SELL: 0.1BTC, SELL: 0.1BTC, BUY: 0.11BTC, SELL: 0.01BTC, SELL: 0.5BTC, BUY: 0.5BTC, BUY: 0.5BTC, SELL: 0.2BTC, BUY: 0.2BTC,"
crypto_balance_str={'BTC': '1.0', 'ETH': 2.9, 'DOGE': '12.0', 'XRP': '100.0', 'SOL': '1.0'} #DELETE
user_id_str='test1'#delete
#
for j in crypto_balance_str.keys():
    if j ==coin:
        a=0
        crp_bal=float(crypto_balance_str[j])
        if crp_bal>how_cc_t:
            crp_bal=crp_bal-how_cc_t
            crp_bal=str(round(crp_bal,3))
            history_=history_str+' TRANSFERRED: '+crp_bal+coin+','
            crypto_balance_str[j]=crp_bal
            query12="update userdata set crypto_balance = (%s), history =(%s) where userid=(%s);"
            val12=(str(crypto_balance_str), history_, user_id_str) 
            cur.execute(query12,val12)
            db.commit()
            a=0

            messagebox.showinfo("SOLD Coin", "Congrats you have sold the coins.")

        else:
            messagebox.showerror("COINS TOO LOW", "You don't have enough coins to Transfer! Try a lower number.")



import mysql.connector 

db = mysql.connector.connect(host='localhost', user='root', password='Arnavcool1', database='crypto')

cur=db.cursor()

#cur.execute("create table userdata ( userid varchar(30) primary key, password varchar(20), name varchar(30), usd_balance int(12),crypto_balance varchar(1600), history varchar(1600))")

#new user

a='test13'
b='test11'
c='ARNAV1 SUMAN'
d=10002
e=''
f=''
a=3.4
b=5.5
print(a-b)

query="INSERT INTO userdata values(%s,%s,%s,%s,%s,%s);"
val=(a,b,c,d,e,f)
cur.execute(query, val)
db.commit()
print('done')
--------------------------------
CREATE DATABASE
cur.execute("create table userdata ( userid varchar(30) primary key, password varchar(20), name varchar(30), usd_balance int(12),crypto_balance varchar(1600), history varchar(1600))")
---------------------------------
SELECT DATA
import mysql.connector 

db = mysql.connector.connect(host='localhost', user='root', password='Arnavcool1', database='crypto')

cur=db.cursor()


query="select userid from userdata;"

cur.execute(query)
row=cur.fetchall()
for i in row:
    print(i)


"""



"""
#first put this on top of page

import mysql.connector 

db = mysql.connector.connect(host='localhost', user='root', password='Arnavcool1', database='crypto')

cur=db.cursor()

#PASSWORD AND USERID


query="select userid from userdata;"

cur.execute(query)
rows=cur.fetchall()

user_id_sql=[] # it stores all userid
user_password_sql=[] #it stores all password used in login screen

for i in rows:
    user_id_sql.append(str(i)[2:-3])


query2="select password from userdata;"
cur.execute(query2)

rows2=cur.fetchall()
for j in rows2:
    user_password_sql.append(str(j)[2:-3])

#GET USER INFORMATION

query="select * from userdata where userid=(%s);"
val=('test2',)
cur.execute(query,val)
rows=cur.fetchall()

for i in rows:
    user_id_str=list(i)[0]
    password_str=list(i)[1]
    name_str=list(i)[2]
    usd_balance_str=list(i)[3]
    crypto_balance_str=list(i)[4]
    history_str=list(i)[5]

#user_id_str password_str name_str usd_balance_str crypto_balance_str history_str
#put entry.get in val as tple

#UPDATE USERDATA


query="update userdata set usd_balance = (%s), crypto_balance = (%s), history =(%s) where userid=(%s);"
val=(int(usd_balance_str),crypto_balance_str,history_str,'test1') #Put accordingly 'test1' is userid of operating user
cur.execute(query,val)

db.commit()
'''

'''
Get Crypto today price

cryptocompare.get_price('BTC', currency='USD', full=True)

to get low, high, opwn volume amount
cryptocompare.get_historical_price_day('BTC', 'USD', limit=any, toTs=datetime.datetime(2021,2,4))

get crypto price on that year
cryptocompare.get_historical_price('XMR', 'EUR', datetime.datetime(2017,6,6))

'''
'''
Get Crypto today price

cryptocompare.get_price('BTC', currency='USD', full=True)

to get low, high, opwn volume amount
cryptocompare.get_historical_price_day('BTC', 'USD', limit=any, toTs=datetime.datetime(2021,2,4))

get crypto price on that year
cryptocompare.get_historical_price('XMR', 'EUR', datetime.datetime(2017,6,6))
'''

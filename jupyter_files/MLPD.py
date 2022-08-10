

import pandas as pd

import warnings
warnings.filterwarnings("ignore")
import collections
from datetime import datetime
def rank_d(data,varia,buckets):

    rank = varia + '_rank'
    data[rank] = 0
    data.sort_values(by = varia, inplace = True)
    data.reset_index(drop = True,inplace = True)
    seperate = len(data) // buckets
    for i in data.index:
        data[rank][i] = min(i // seperate + 1,buckets)
    return data
class Context:
    def __init__(self,cash,start_date, end_date,select_trading):
        self.cash = cash
        self.start_date = start_date
        self.end_date = end_date
        self.positions = {}
        self.benchmark = None
        self.data = select_trading[(select_trading['交易日期']<= end_date) &\
                                    (select_trading['交易日期']>= start_date)
                                    ]
        self.str_frame = {}
        self.coupon = 0
        self.price_profit = 0    
    
def pnl_daily(bond_code, today,context,dataset):
   context.positions[bond_code] = context.positions.get(bond_code,0)*(1+dataset[dataset['代码'] == bond_code]['涨跌幅'].values[0]/100)
   
class Coupon_bond:
    def get_price(self,coupon,face_value,int_rate,years,freq):
        total_coupons_pv = self.get_coupons_pv(coupon,int_rate,years,freq)
        face_value_pv    = self.get_face_value_pv(face_value,int_rate,years)
        result           = total_coupons_pv + face_value_pv
        return result
        
    @staticmethod
    def get_face_value_pv(face_value,int_rate,years):
        fvpv = face_value / (1 + int_rate)**years
        return fvpv
    
    def get_coupons_pv(self,coupon,int_rate,years,freq=1):
        pv = 0
        for period in range(int(years * freq)):
            pv += self.get_coupon_pv(coupon,int_rate,period+1,freq)
        return pv
    
    @staticmethod
    def get_coupon_pv(coupon, int_rate,period,years,freq):
        pv = coupon / (1+int_rate/freq)**(period + ((years*freq)%(360/freq)))
        return pv
    
    def get_ytm(self,bond_price,face_value,coupon,years,freq=1,estimate=0.05):        
        return ((coupon + (face_value-bond_price)/ years*freq)) / ((face_value + bond_price)/2)
    
df= pd.read_excel('/Users/weiliu/Desktop/Bond/data/财务数据0612.xlsx', index_col = None)#财务数据0519\财务数据1011\0612
df['year'] = df['year'] + 1
pred_pd1 = pd.read_excel('/Users/weiliu/Desktop/Bond/ensemble_rf_pd_0612.xlsx')#0513
tradata1 = pd.read_excel("/Users/weiliu/Desktop/Bond/data/bond_trade0806(上市企业发行债券).xlsx")
tradata1.columns = ['交易日期',	'代码',	'名称',	'最高价',	'最低价',	'收盘净价',	'收盘全价',	'涨跌幅',	'成交额',	'均价',	'COMPANYCODE']
tradata1['交易日期'] = pd.to_datetime(tradata1['交易日期'])
default_list = pd.read_excel('/Users/weiliu/Desktop/Bond/债券违约事件一览.xls') 

# bond_info.columns






bond_info1 = pd.read_excel('/Users/weiliu/Desktop/Bond/bond_info0704update.xlsx')

remove = ['同业存单','可分离转债存债','可交换债','可转换债券','定向工具','利率债','可转债','可交换债券','资产支持证券','政府支持机构债']
#bond_info = bond_info1[(~bond_info1['东财债券一级分类(2021)'].isin(remove))&(~bond_info1['东财债券二级分类(2021)'].isin(remove))&(bond_info1['是否可转债']=='否')&(bond_info1['募集方式']=='公募债券')]#&(bond_info1['可回售性']=='否'&(bond_info1['可赎回性']=='否'))&&(bond_info1['募集方式']=='公募债券')
bond_info = bond_info1[(~bond_info1['EM1TYPE2021'].isin(remove))&(~bond_info1['EM2TYPE2021'].isin(remove))&(bond_info1['IS_CONVERTIBLE_BOND']=='否')&(bond_info1['ISSUETYPE']==1)]#&(bond_info1['可回售性']=='否'&(bond_info1['可赎回性']=='否'))&&(bond_info1['募集方式']=='公募债券')
#bond_info = bond_info1[(~bond_info1['EM1TYPE2021'].isin(remove))&(~bond_info1['EM2TYPE2021'].isin(remove))&(bond_info1['IS_CONVERTIBLE_BOND']=='否')&(bond_info1['ISSUETYPE']==1)]
#bond_info[bond_info['SECURITYNAME'] == '21长电CP002']
#bond_info = bond_info[(bond_info['到期日期'] >= ' 2022-12-31') & (bond_info['发行起始日期'] <= '2020-01-01')]
# bond_info.columns
# bond_info = bond_info.dropna(subset=['每年付息日'])
# bond_dict ={}
# for i in bond_info.index:
#     bond_dict[bond_info['证券代码'][i]] = {}
#     bond_dict[bond_info['证券代码'][i]]['债券发行时面值'] = bond_info['债券发行时面值(元)'][i]
#     bond_dict[bond_info['证券代码'][i]]['票面利率'] = bond_info['票面利率(当期)'][i]
#     bond_dict[bond_info['证券代码'][i]]['到期日期'] = pd.Timestamp(bond_info['到期日期'][i])
#     bond_dict[bond_info['证券代码'][i]]['每年付息次数']= len(list(bond_info['每年付息日'][i]))
    
# #bond_info[(bond_info['到期日期'] >= '2022-06-31')]['到期日期']
    
# """
# 交易记录 和 债券信息 匹配current yield
# """
# code_b = {bond_info['证券代码'][i]:bond_info['公司代码'][i] for i in bond_info.index}

# tradata1['company_code'] = '' 

# for i in tradata1.index:
#     try:
#         tradata1['company_code'][i] = str(int(code_b[tradata1['代码'][i]]))
#     except:
#         continue
    
# tradata1 = tradata1.dropna()
bond_info.columns
bond_info = bond_info.dropna(subset=['PAYDAY'])
bond_dict ={}
for i in bond_info.index:
    bond_dict[bond_info['SECURITYCODE'][i]] = {}
    bond_dict[bond_info['SECURITYCODE'][i]]['债券发行时面值'] = bond_info['PAR'][i]
    bond_dict[bond_info['SECURITYCODE'][i]]['票面利率'] = bond_info['COUPONRATECURRENT'][i]
    bond_dict[bond_info['SECURITYCODE'][i]]['到期日期'] = pd.Timestamp(bond_info['MRTYDATE'][i])
    bond_dict[bond_info['SECURITYCODE'][i]]['每年付息次数']= len(list(bond_info['PAYDAY'][i]))
    
#bond_info[(bond_info['到期日期'] >= '2022-06-31')]['到期日期']
    
"""
交易记录 和 债券信息 匹配current yield
"""
code_b = {bond_info['SECURITYCODE'][i]:bond_info['COMPANYCODE'][i] for i in bond_info.index}

tradata1['company_code'] = '' 

for i in tradata1.index:
    try:
        tradata1['company_code'][i] = str(int(code_b[tradata1['代码'][i]]))
    except:
        continue
    
tradata1 = tradata1.dropna()
#tradata1.to_excel(r'/Users/weiliu/Desktop/Bond/data/bond_trade_try0301.xlsx')
#tradata['merton_pd'] = 0.0
tradata1['交易日期']
# len(set(bond_info1[bond_info1['是否上市公司'] == '是']['发行人中文名称']))

# com_name = {'listed_name': list(set(bond_info1[bond_info1['是否上市公司'] == '是']['发行人中文名称'])),
#             'listed_code':  list(set(bond_info1[bond_info1['是否上市公司'] == '是']['公司代码']))}
# pd.DataFrame(data = com_name).to_excel(r'/Users/weiliu/Desktop/Bond/listed_company.xlsx')

"""
连接 债券交易记录 和 股市交易记录， 看债券交易记录里 每天的 merton pd
"""
# x = tradata1.merge(merton, how = 'left', left_on=['company_code','交易日期'], right_on = ['company_code','tradedate'])

# tradata = x.dropna(subset = ['pd_naive'])
tradata = tradata1





def prepare_portfolio_ml(year,buck_nums,remove_threshold,yield_threshold,start_date,end_date,maturity,back_day):#
#选取2020年的数据
    pred_pd = pred_pd1[pred_pd1['year'] == year]
    print(start_date,end_date)

    pred_pd = rank_d(pred_pd,'PD',buck_nums)
    
    #print(pred_pd)
    #pred_pd[pred_pd['PD_rank'] == 7]
    
    start_3m = pd.Timestamp(start_date) - pd.Timedelta(days = back_day)
    end_3m = pd.Timestamp(start_date) #start_3m + pd.Timedelta(days = 30)
    
    
    # data = pred_pd.merge(df,left_on = ['公司代码','year'],right_on = ['公司代码','year'])
    
    # data = data.dropna(subset = ['PD'])
    
    data = pred_pd.merge(df,left_on = ['公司代码','year'],right_on = ['公司代码','year'],suffixes=('', '_right'))
    
    data = data.dropna(subset = ['PD'])
    
    bond_info.columns
    
    data = data[['公司代码','发行人中文名称','PD','year','m_score','PD_rank']]#

    
    tra_2020 = tradata[(tradata['交易日期'] >= start_date) & (tradata['交易日期'] <= end_date)]
    #以起始点前三个月的 进行筛选 merton pd, 选出每个公司的 merton mean
    tra_3m = tradata[(tradata['交易日期'] >= start_3m) & (tradata['交易日期'] <= end_3m)]
    #print("# of bonds that have trading records in last 3 months:",len(set(tra_3m['代码'])))
    #print(data)
    #print(len(set(tra_3m['company_code'])))
    merton_mean = tra_3m.groupby(by = 'company_code').mean().reset_index()
    #print(merton_mean)
    select_bond = data.merge(bond_info,how = 'left', left_on = '公司代码',right_on = 'COMPANYCODE',suffixes=('', '_right'))
    select_bond = select_bond.dropna(subset=['COMPANYCODE'])
    #print(select_bond.columns)
    #select_bond2 = copy.copy(bond_info)
    select_bond['COMPANYCODE'] = select_bond['COMPANYCODE'].astype(int).astype(str)
    select_bond = select_bond[select_bond['ISLISTED'] == '是']
    #select_bond = rank_d(select_bond,'PD',10)
    #select_bond2['公司代码'] = select_bond2['公司代码'].astype(int).astype(str)
    print("# of companies that both have ML results and trading records in last 3 months:",len(set(select_bond['COMPANYCODE'])))
    select_bond = select_bond.merge(merton_mean,how ='left',left_on = 'COMPANYCODE',right_on ='company_code',suffixes=( '_right',''))#select_bond company_code
    print("# of companies that both have ML results and trading records in last 3 months:",len(set(select_bond['COMPANYCODE'])))
    #print("# of bonds after filter bonds type:",len(set(select_bond['证券代码'])))

    #select_bond2 = merton_mean.merge(select_bond2,how ='left',left_on = 'company_code',right_on ='公司代码')
    bond_info.columns
    #select_bond = select_bond[['证券代码','证券名称','公司代码' ,'发行人中文名称_y','year','pd_naive','m_score','每年付息日','票面利率(当期)' ,'年付息频率(数字)']]
    select_bond = select_bond[['SECURITYCODE','SECURITYNAME','COMPANYCODE' ,'COMPANYNAME','PD','PAYDAY','COUPONRATECURRENT' ,'PAYPERYEAR','PD_rank','PAR']]
    #select_bond = select_bond[select_bond['是否上市公司'] == '是']
    
    len(set(select_bond['SECURITYCODE']))
    #len(set(buckets_3m['证券代码']))
    #buckets_3m.columns
    
    #merton_mean.merge(select_bond,how ='left',left_on = 'company_code',right_on ='公司代码',suffixes=('', '_right')).columns
    
    
    
    print("For top {:.0%} highest yield combination: ".format(yield_threshold))
    
    #buckets = rank_d(select_bond,'PD',10)
    #select_buckets = {k: buckets[buckets['PD_rank'] == k] for k in set(buckets.PD_rank)}
    #choose all the top 50% highest yield  bonds from  each bucket
    
    #计算过去3个月内 current yield 平均，并选取最高的50%
    buckets_3m = select_bond.merge(tra_3m,left_on = 'SECURITYCODE', right_on = '代码')
    #print(buckets_3m)
    # print(tra_3m)
    # print(select_bond)
    #print(buckets_3m)
    print("# of bonds that both have ML results and trading records in last 3 months:",len(set(buckets_3m['SECURITYNAME'])))
    #print("# of bonds after filter bonds type:",len(set(buckets_3m['代码'])))
    buckets_3m['c_yield'] = 0.0 
    for i in buckets_3m.index:
        buckets_3m['c_yield'][i] = bond_dict[buckets_3m['代码'][i]]['票面利率'] / buckets_3m['收盘净价'][i]
    
    #  
    #Maturity
    buckets_3m['Maturity'] = 0.0
    for i in buckets_3m.index:
        buckets_3m['Maturity'][i] =  (bond_dict[buckets_3m['代码'][i]]['到期日期'] - pd.Timestamp(buckets_3m['交易日期'][i])).days / 365
    buckets_3m = buckets_3m[buckets_3m['Maturity']>= maturity]
    #buckets_3m.groupby(by = '证券代码').mean().loc['031780001.IB']
    #buckets = rank_d(buckets_3m.groupby(by = '证券代码').mean().reset_index(),'PD',10)
    select_buckets = {k: buckets_3m[buckets_3m['PD_rank'] == k].groupby(by = 'SECURITYCODE').mean() for k in set(buckets_3m.PD_rank)}
    #print(buckets_3m.PD_rank)
    print("# of bonds after remove Maturity < 1:",len(set(buckets_3m['SECURITYNAME'])))
    print("# of bonds after remove Maturity < 1:",len(set(buckets_3m['SECURITYNAME'])))
    
    # yield_buckets = {k: select_buckets[k][select_buckets[k]['c_yield'] >= select_buckets[k]['c_yield'].quantile(0.5)] for k in select_buckets.keys()}
    if remove_threshold == 0:
        yield_buckets = {k: select_buckets[k] for k in list(select_buckets.keys())[:]}
    else:
        yield_buckets = {k: select_buckets[k] for k in list(select_buckets.keys())[:-1*remove_threshold]}
    
    #print(yield_buckets)
    portfolio_frame = pd.DataFrame()
    for i in yield_buckets:
        portfolio_frame = pd.concat([portfolio_frame,yield_buckets[i]])
    #print(portfolio_frame.columns)
    print("# of bonds after remove last 3 ML buckets:",len(portfolio_frame[~portfolio_frame.index.duplicated(keep='first')].index))

    yield_20 = portfolio_frame[portfolio_frame['c_yield'] >= portfolio_frame['c_yield'].quantile(1-yield_threshold)]
    #yield_20.reset_index(inplace = True)
    yield_20 = yield_20[~yield_20.index.duplicated(keep='first')]
    print("# of bonds after filter yield:",len(set(yield_20.index)))
    performance['start_date'].append(start_date)
    performance['end_date'].append(end_date)
    performance['Yield in (top)'].append(str(round(yield_threshold*100,0)) + '%')
    # #portfolio
 
    # portfolio_array = [yield_buckets[k].index for k in yield_buckets.keys()]
    #portfolio_array = [yield_buckets[k].index for k in yield_buckets.keys()]
    portfolio_list = []
    
    #list(yield_buckets[5].index.values)
    #print(yield_20)
    for i in yield_20.index:
        portfolio_list.append(i)
        yield_dict[i] = yield_20.loc[i]['c_yield']
    
    print("Number of bonds in portfolio",len(portfolio_list))
    
    """
    计算portfolio 平均 yield 和 maturity 
    """
    
    average_yield = buckets_3m[buckets_3m['SECURITYCODE'].isin(portfolio_list)].groupby(by = 'SECURITYCODE').mean()['c_yield'].mean()
    average_maturity = buckets_3m[buckets_3m['SECURITYCODE'].isin(portfolio_list)].groupby(by = 'SECURITYCODE').mean()['Maturity'].mean()
    
    print("average_yield:",str(round(average_yield*100,2))+"%")
    print("average_maturity:",round(average_maturity),2)
    performance['avg_yield'].append(str(round(average_yield*100,2))+"%")
    performance['avg_maturity'].append(round(average_maturity,2))    
    tra_ = tra_2020[tra_2020['代码'].isin(portfolio_list)]
    tra_ = tra_.merge(select_bond, left_on = '代码', right_on = 'SECURITYCODE')
    performance['# bonds In Portfiolio'].append(len(portfolio_list))
    yield_20.reset_index(inplace = True)
    return tra_2020, tra_,yield_20

def run_system_ml(start_date,end_date,initial_fund):
    CASH = initial_fund
    START_DATE = start_date
    END_DATE = end_date
    grf_select = Context(CASH,START_DATE,END_DATE, tra_)
    
    grf_select.data.groupby(by = '代码').count()['交易日期'].min()
    select_ID = set(tra_['代码'])
    amount = grf_select.cash / len(select_ID)#len(grf_select.data.groupby(by = '代码').count()[grf_select.data.groupby(by = '代码').count()['交易日期'] >= 10])
    
    print("number of bonds actually traded in the backtesting:", len(select_ID))
    performance['# bonds Actually Traded'].append(len(select_ID))   
    
    
    for i in select_ID:
        #if i != '101800353.IB' and len(grf_select.data[grf_select.data['代码'] == i]) >=10:
        #建仓
        grf_select.positions[i] = amount
        
        grf_select.cash -= amount
        #起始价格
        target_data = grf_select.data[grf_select.data['代码'] == i]
        #print(target_data)
        target_data = target_data.sort_values(by = ['交易日期'],ascending=False)
        start_price = target_data.iloc[-1]['收盘净价']
        #结束价格
        end_price = target_data.iloc[0]['收盘净价']
        #计算PNL
        pnl = (end_price - start_price) / (start_price)
        #print(i,pnl)
        #计算最终持有价值
        grf_select.positions[i] *= (1+pnl)
        #计算coupon
        time = target_data.iloc[-1]['PAYPERYEAR']
        rate = target_data.iloc[-1]['COUPONRATECURRENT']
        #coupon = time * rate * 100
        #收益加上coupon
        # if i in list(default_data['代码']):
        #     coupon_profit = 0
        #     print("default:",i)
        # else:
        number = amount / start_price
        coupon_profit = number * time * rate * (pd.Timestamp(end_date) - pd.Timestamp(target_data.iloc[-1]['交易日期'])).days / 365
        grf_select.coupon += coupon_profit
        grf_select.price_profit += grf_select.positions[i] 
        #print(target_data['收盘净价'].max(),target_data['收盘净价'].min())
        #print(target_data.iloc[-1]['证券名称'],start_price,end_price)
        #print(target_data.columns)
        portframe['company_code'].append(target_data.iloc[-1]['COMPANYCODE_x'])
        portframe['bond_name'].append(target_data.iloc[-1]['SECURITYNAME'])
        portframe['bond_code'].append(target_data.iloc[-1]['代码'])
        portframe['company_name'].append(target_data.iloc[-1]['COMPANYNAME'])
        portframe['start_price'].append(start_price)
        portframe['end_price'].append(end_price)
        portframe['ml_pdrank'].append(target_data.iloc[-1]['PD_rank'])
        portframe['price_spread'].append(end_price - start_price)
        portframe['coupon_rate'].append(target_data.iloc[-1]['COUPONRATECURRENT'])
        portframe['yield'].append(target_data.iloc[-1]['COUPONRATECURRENT'])
        portframe['matutiry'].append((pd.Timestamp(bond_info[bond_info['SECURITYCODE'] == i]['MRTYDATE'].values[0]) - pd.Timestamp(end_date)).days / 365)
        portframe['ytm'].append(calculator.get_ytm(bond_price= end_price,\
                                 face_value= bond_info[bond_info['SECURITYCODE'] == i]['PAR'].values[0],\
                                     
                                     coupon=target_data.iloc[-1]['COUPONRATECURRENT'],\
                                         years= (pd.Timestamp(bond_info[bond_info['SECURITYCODE'] == i]['MRTYDATE'].values[0]) - pd.Timestamp(end_date)).days / 365,\
                                             freq=target_data.iloc[-1]['PAYPERYEAR']))
        bond_info.MRTYDATE
    print(grf_select.coupon, grf_select.price_profit,grf_select.cash)
    
    total = grf_select.coupon + grf_select.price_profit + grf_select.cash
    grf_select.positions
    print("The return of the portfolio is ", "%.3f"%((total - CASH) / CASH)) #0.03851823807476366
    performance['Total return'].append(str(round(100 * (total - CASH) / CASH,3)) + "%")
    performance['remove_threshold'].append(remove_threshold)

    print('===================================================================')
    return grf_select

df.columns

p=0.2
calculator = Coupon_bond()
for tp in ['ml','benchmark']:
    if tp == 'ml':
        performance =  collections.defaultdict(list)
    for th in [1,2,3]:
        if tp == 'ml':
            remove_threshold = th
        elif tp == 'benchmark':
            performance =  collections.defaultdict(list)
            remove_threshold = 0
        
        
        
        
        start_date = '2022-04-30'
        end_date = '2022-08-07'
        initial_fund = 100000000
        today = datetime.today().strftime('%Y-%m-%d')
        date_range = start_date + '-' + end_date        
        for p in [1.0,0.5,0.4,0.3,0.25,0.2]:#1
            yield_dict = {}
            parameters = {
            'year': 2022,
            'buck_nums': 10,
            'remove_threshold': remove_threshold,
            'yield_threshold': p,
            'start_date':start_date,
            'end_date':end_date,
            'maturity':1,
            'back_day':90
            # 'start_3m':'2020-01-31',
            # 'end_3m':'2020-04-29'
            }
            
            portframe = {}
            portframe['company_code'] = []
            portframe['bond_name'] = []
            portframe['bond_code'] = []
            portframe['company_name'] = []
            portframe['start_price'] = []
            portframe['end_price'] = []
            portframe['ml_pdrank'] = []
            portframe['price_spread'] = []
            portframe['coupon_rate'] = []
            portframe['yield'] = []
            portframe['matutiry'] = []
            portframe['ytm'] = []
            #data = df[df['year'] == 2020]
        
            tra_year, tra_, portfolio = prepare_portfolio_ml(**parameters)
            
            grf_select = run_system_ml(start_date,end_date,initial_fund)
            if p ==1.0 and tp =='benchmark':
                
                portframe_fram = pd.DataFrame(data = portframe)
                portframe_fram.drop_duplicates(inplace = True)
                portframe_fram['yield']= 0.0
                for i in portframe_fram.index:
                    portframe_fram['yield'][i]= yield_dict[portframe_fram.loc[i]['bond_code']]
                portframe_fram['Top_yield_percentile'] = portframe_fram['yield'].rank(pct=True,ascending=False)
                #portframe_fram = portframe_fram[portframe_fram['bond_code'].isin(list(default_data['代码']))]
                portframe_fram
                portframe_fram.to_excel(r"/Users/weiliu/Desktop/Bond/backtesting result/%s/new %s dataframe %s.xlsx"%(today,tp,date_range))
            
        
    

        performance_fram = pd.DataFrame(data = performance)
        performance_fram.to_excel(r"/Users/weiliu/Desktop/Bond/backtesting result/%s/new %s yearly %s.xlsx"%(today,tp,date_range))   
    
    #grf_select.data.to_excel(r"/Users/weiliu/Desktop/Bond/backtesting result/加入期权 2021/portfolio bond ml benchmark.xlsx")
    
    
    
    
    
        #default_data = default_list[(default_list['违约日期'] >= pd.Timestamp(start_date)) & (default_list['违约日期']<= pd.Timestamp(end_date))]
    
    
        portframe_fram = pd.DataFrame(data = portframe)
        portframe_fram.drop_duplicates(inplace = True)
        portframe_fram['yield']= 0.0
        for i in portframe_fram.index:
            portframe_fram['yield'][i]= yield_dict[portframe_fram.loc[i]['bond_code']]
        portframe_fram['Top_yield_percentile'] = portframe_fram['yield'].rank(pct=True,ascending=False)
        #portframe_fram = portframe_fram[portframe_fram['bond_code'].isin(list(default_data['代码']))]
        portframe_fram
        portframe_fram.to_excel(r"/Users/weiliu/Desktop/Bond/backtesting result/%s/new remove %s %s dataframe %s.xlsx"%(today,th,tp,date_range))



# """
# """


# for tp in ['ml','benchmark']:
#     if tp == 'ml':
#         performance =  collections.defaultdict(list)
#     for th in [1,2,3]:
#         if tp == 'ml':
#             remove_threshold = th
#         elif tp == 'benchmark':
#             performance =  collections.defaultdict(list)
#             remove_threshold = 0
        
        
        
        
#         start_date = '2022-04-30'
#         end_date = '2023-04-30'
#         initial_fund = 100000000
#         today = datetime.today().strftime('%Y-%m-%d')
#         date_range = start_date + '-' + end_date        
#         for p in [1.0,0.5,0.4,0.3,0.25,0.2]:#1
#             yield_dict = {}
#             parameters = {
#             'year': 2022,
#             'buck_nums': 10,
#             'remove_threshold': remove_threshold,
#             'yield_threshold': p,
#             'start_date':start_date,
#             'end_date':end_date,
#             'maturity':1,
#             'back_day':90
#             # 'start_3m':'2020-01-31',
#             # 'end_3m':'2020-04-29'
#             }
            
#             portframe = {}
#             portframe['company_code'] = []
#             portframe['bond_name'] = []
#             portframe['bond_code'] = []
#             portframe['company_name'] = []
#             portframe['start_price'] = []
#             portframe['end_price'] = []
#             portframe['ml_pdrank'] = []
#             portframe['price_spread'] = []
#             portframe['coupon_rate'] = []
#             portframe['yield'] = []
#             #data = df[df['year'] == 2020]
        
#             tra_year, tra_, portfolio = prepare_portfolio_ml(**parameters)
#             if tp == 'ml':
#                 portframe_fram = portfolio
#                 portframe_fram = portframe_fram[['证券代码','PD','PD_rank','c_yield', 'Maturity']]
#                 portframe_fram.columns = ['bond_code','PD','PD_rank','yield', 'Maturity']
#                 portframe_fram.to_excel(r"/Users/weiliu/Desktop/Bond/backtesting result/%s/new MLM remove %s %s top %s percent dataframe %s.xlsx"%(today,th,tp,str(p*100),date_range))
            

# list(portfolio.columns)

# tra_2020 = tradata[(tradata['交易日期'] >= start_date) & (tradata['交易日期'] <= end_date)]
# tra_2020.columns

# portframe_fram.columns


# for i in portframe:   
#     print(len(portframe[i]))
    
# portframe_fram['']
    
    
# code_b['149061.SZ']
# bond_info[bond_info['证券代码'] == '101901406.IB']['到期日期']
    
# start_date = '2021-04-30'
# end_date = '2022-04-30' 
# pred_pd = pred_pd1[pred_pd1['year'] == 2021]
# print(start_date,end_date)

# pred_pd = rank_d(pred_pd,'PD',10)


# #pred_pd[pred_pd['PD_rank'] == 7]

# # start_3m = pd.Timestamp(start_date) - pd.Timedelta(days = back_day)
# # end_3m = pd.Timestamp(start_date) #start_3m + pd.Timedelta(days = 30)


# data = pred_pd.merge(df,left_on = ['公司代码','year'],right_on = ['公司代码','year'])

# data = data.dropna(subset = ['PD'])

# data = pred_pd.merge(df,left_on = ['公司代码','year'],right_on = ['公司代码','year'],suffixes=('', '_right'))

# data = data.dropna(subset = ['PD'])

# bond_info.columns

# data = data[['公司代码','发行人中文名称','PD','year','m_score','PD_rank']]#


# data

# select_bond = data.merge(bond_info,how = 'left', on = '公司代码',suffixes=('', '_right'))

# select_bond = select_bond[select_bond['证券代码'].isin(list(default_data['代码']))]

# # select_bond.to_excel(r"/Users/weiliu/Desktop/Bond/backtesting result/%s/%s dataframe defaultd %s.xlsx"%(today,tp,date_range))
# select_bond.columns

# select_bond
# tra_

# tra_2020 = tradata[(tradata['交易日期'] >= start_date) & (tradata['交易日期'] <= end_date)]
# tra_2020.columns
# select_ID = set(select_bond['证券代码'])
# select_trading = select_bond.merge(tra_2020,how='left',left_on='证券代码',right_on = '代码')
# portframe =  collections.defaultdict(list)
# for i in select_ID:
#         #if i != '101800353.IB' and len(grf_select.data[grf_select.data['代码'] == i]) >=10:
#         #建仓
#     #grf_select.positions[i] = amount
    
#     #grf_select.cash -= amount
#     #起始价格
#     try:
#         target_data = select_trading[select_trading['代码'] == i]
#         #print(target_data)
#         target_data = target_data.sort_values(by = ['交易日期'],ascending=False)
#         start_price = target_data.iloc[-1]['收盘净价']
#         #结束价格
#         end_price = target_data.iloc[0]['收盘净价']
#         #计算PNL
#         pnl = (end_price - start_price) / (start_price)
#         #print(i,pnl)
#         #计算最终持有价值
#         #grf_select.positions[i] *= (1+pnl)
#         #计算coupon
#         time = target_data.iloc[-1]['年付息频率(数字)']
#         rate = target_data.iloc[-1]['票面利率(当期)']
#         #coupon = time * rate * 100
#         #收益加上coupon
#         # if i in list(default_data['代码']):
#         #     coupon_profit = 0
#         #     print("default:",i)
#         # else:
#         #number = amount / start_price
#         #coupon_profit = number * time * rate * (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365
#         #grf_select.coupon += coupon_profit
#         #grf_select.price_profit += grf_select.positions[i] 
#         portframe['company_code'].append(target_data.iloc[-1]['公司代码'])
#         portframe['bond_name'].append(target_data.iloc[-1]['证券名称'])
#         portframe['bond_code'].append(target_data.iloc[-1]['代码'])
#         portframe['company_name'].append(target_data.iloc[-1]['发行人中文名称'])
#         portframe['start_price'].append(start_price)
#         portframe['end_price'].append(end_price)
#         portframe['ml_pdrank'].append(target_data.iloc[-1]['PD_rank'])
#         portframe['price_spread'].append(end_price - start_price)
    
#     except:
#         continue
# portframe_fram = pd.DataFrame(data = portframe)
# portframe_fram.to_excel(r"/Users/weiliu/Desktop/Bond/backtesting result/default_spread 2021.xlsx")
    
# pred_pd = pred_pd1[pred_pd1['year'] == 2022]
# print(start_date,end_date)

# pred_pd = rank_d(pred_pd,'PD',buck_nums)


# #pred_pd[pred_pd['PD_rank'] == 7]

# start_3m = pd.Timestamp(start_date) - pd.Timedelta(days = back_day)
# end_3m = pd.Timestamp(start_date) #start_3m + pd.Timedelta(days = 30)


# # data = pred_pd.merge(df,left_on = ['公司代码','year'],right_on = ['公司代码','year'])

# # data = data.dropna(subset = ['PD'])

# data = pred_pd.merge(df,left_on = ['公司代码','year'],right_on = ['公司代码','year'],suffixes=('', '_right'))

# data = data.dropna(subset = ['PD'])

# bond_info.columns

# data = data[['公司代码','发行人中文名称','PD','year','m_score','PD_rank']]#
pred_pd = pred_pd1[pred_pd1['year'] == 2022]
print(start_date,end_date)

pred_pd = rank_d(pred_pd,'PD',100)


#pred_pd[pred_pd['PD_rank'] == 7]
start_date = '2022-04-30'
end_date = '2022-06-30'
start_3m = pd.Timestamp(start_date) - pd.Timedelta(days = 90)
end_3m = pd.Timestamp(start_date) #start_3m + pd.Timedelta(days = 30)


# data = pred_pd.merge(df,left_on = ['公司代码','year'],right_on = ['公司代码','year'])

#data = data.dropna(subset = ['PD'])

data = pred_pd.merge(df,left_on = ['公司代码','year'],right_on = ['公司代码','year'],suffixes=('', '_right'))

data = data.dropna(subset = ['PD'])

bond_info.columns

data = data[['公司代码','发行人中文名称','PD','year','m_score','PD_rank']]#


tra_2020 = tradata[(tradata['交易日期'] >= start_date) & (tradata['交易日期'] <= end_date)]
#以起始点前三个月的 进行筛选 merton pd, 选出每个公司的 merton mean
tra_3m = tradata[(tradata['交易日期'] >= start_3m) & (tradata['交易日期'] <= end_3m)]
#print("# of bonds that have trading records in last 3 months:",len(set(tra_3m['代码'])))
#print(data)  
merton_mean = tra_3m.groupby(by = 'company_code').mean().reset_index()

select_bond = data.merge(bond_info,how = 'left', left_on = '公司代码',right_on = 'COMPANYCODE',suffixes=('', '_right'))
select_bond.dropna(subset =['PD'])
print(select_bond.columns)
#print(len(set(select_bond[select_bond['到期日期']>='2022-04-30']['证券名称'])))

len(set(tra_3m.merge(select_bond, how='left',right_on = 'SECURITYCODE', left_on = '代码').dropna(subset =['PD'])['代码']))
len(set(select_bond.merge(tra_3m, how='left',left_on = 'SECURITYCODE', right_on = '代码').dropna(subset =['PD'])['代码']))
len(set(tra_3m['company_code']))

pred_pd['公司代码'] = pred_pd['公司代码'].astype(int).astype(str)
tra_3m = tradata1[(tradata1['交易日期'] >= start_3m) & (tradata1['交易日期'] <= end_3m)]
tra_3m.to_excel(r'/Users/weiliu/Desktop/Bond/data/bond_trade_try_3m.xlsx')
tra_3m.merge(pred_pd,how = 'left', left_on = 'company_code',right_on = '公司代码').to_excel(r'/Users/weiliu/Desktop/Bond/data/bond_trade_try_3m_pd.xlsx')
tra_3m['company_code']
tra_3m.merge(pred_pd,how = 'left', left_on = 'company_code',right_on = '公司代码').columns
try0706 = tra_3m.merge(pred_pd,how = 'left', left_on = 'company_code',right_on = '公司代码').merge(bond_info1,how = 'left', left_on = '代码',right_on = 'SECURITYCODE')

try0706[['交易日期', '代码', '名称', '最高价', '最低价', '收盘净价', '收盘全价', '涨跌幅', '成交额', '均价','COMPANYCODE_x', 'company_code', 'Unnamed: 0', 'index', '公司代码', '发行人中文名称',
       'year', 'PD', 'PD_rank','EM1TYPE2021','EM2TYPE2021','IS_CONVERTIBLE_BOND','ISSUETYPE']].drop_duplicates(subset=['代码']).to_excel(r'/Users/weiliu/Desktop/Bond/data/bond_trade_try_3m_pd0706-1.xlsx')



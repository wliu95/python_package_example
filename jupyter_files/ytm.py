class Coupon_bond:
    """
    bond_price: 债券最新的收盘价
    face_value: 票面价格，对应bond_info 中 par 字段
    coupon: 对应bond_info 中 COUPONRATECURRENT 字段
    years： Maturity， （到期日期 - 最新收盘价对应日期）365， 到期日期 对应 bond_info 中 MRTYDATE 字段
    freq: 对应 bond_info 中 PAYPERYEAR 字段
    """
    
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
        "用这个公式计算ytm"
        return ((coupon + (face_value-bond_price)/ years*freq)) / ((face_value + bond_price)/2)

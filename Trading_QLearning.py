import math
import talib
import numpy as np
import pandas as pd
import random as rand
from collections import deque
import datetime as dt
from logbook import Logger, StreamHandler, StderrHandler
import sys
from collections import deque
import warnings
warnings.filterwarnings("ignore")
from zipline.api import order_target, record, symbols, sid, date_rules, time_rules, get_order, cancel_order, cancel_policy
from zipline.api import schedule_function, order, history, set_benchmark, order_target_percent, set_slippage, set_commission, commission

import matplotlib.pyplot as plt

class QLearner(object):
    def author(self):
        return 'ywang3134'  

    def __init__(self, \
                 num_states=100, \
                 num_actions=4, \
                 alpha=0.2, \
                 gamma=0.9, \
                 rar=0.5, \
                 radr=0.99, \
                 dyna=0, \
                 verbose=False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.num_states_visit = set([])
        self.QTable = np.random.uniform(-1, 1, size=[num_states, num_actions])
        self.TTable = np.ones(shape=[num_states, num_actions, num_states]) / (num_states + 0.0)
        self.TCTable = np.random.uniform(0.00001, 0.00001, size=[num_states, num_actions, num_states])
        self.RTable = np.random.uniform(-1, 1, size=[num_states, num_actions])
        self.FlagTable = np.zeros(shape=[num_states, num_actions])
        self.visit = {}

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s

        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
            return action

        action, reward = max(enumerate(self.QTable[s]), key=lambda x: x[1])

        if self.verbose: print "s =", s, "a =", action
        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        action, reward = max(enumerate(self.QTable[s_prime]), key=lambda x: x[1])

        self.QTable[self.s][self.a] = (1 - self.alpha) * self.QTable[self.s][self.a] + \
                                      self.alpha * (r + self.gamma * self.QTable[s_prime][action])

        if self.dyna > 0:
            self.TCTable[self.s][self.a][s_prime] += 1
            self.TTable[self.s][self.a][:] = self.TCTable[self.s][self.a][:] / self.TCTable[self.s][self.a][:].sum()

            self.RTable[self.s][self.a] = (1 - self.alpha) * self.RTable[self.s][self.a] + self.alpha * r

            c = self.TTable[self.s][self.a].tolist();
            self.FlagTable[self.s][self.a] = c.index(max(c))

            self.visit[self.s] = set()
            self.visit[self.s].add(self.a)

            # dynaqno = self.dyna
            dynaqno = len(self.visit)
            s_dq = np.random.choice(list(self.visit.keys()), size=dynaqno)
            action_dq = np.random.choice(self.num_actions, size=dynaqno)
            s_prime_dq = np.array(self.FlagTable)[s_dq, action_dq]
            r_dq = np.array(self.RTable)[s_dq, action_dq]

            for id, s_dq_iteration in enumerate(s_dq):
                action, reward = max(enumerate(self.QTable[s_prime_dq[id]]), key=lambda x: x[1])

                self.QTable[s_dq_iteration][action_dq[id]] = (1 - self.alpha) * self.QTable[s_dq_iteration][
                    action_dq[id]] + self.alpha * (r_dq[id] + self.gamma * self.QTable[s_prime_dq[id]][action])

        self.s = s_prime
        self.rar = self.rar * self.radr

        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
            self.a = action
            return action

        action, reward = max(enumerate(self.QTable[self.s]), key=lambda x: x[1])

        self.a = action
        if self.verbose: print "s =", s_prime, "a =", action, "r =", r
        return action
 
 
class StrategyLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.Qiter = 200
        self.base_N = 7
        self.no_indi = 4
        self.no_act = 3
        self.no_days = 1
        self.learner = QLearner(num_states=pow(self.base_N, self.no_indi) * self.no_act, \
                                num_actions=3, \
                                alpha=0.2, \
                                gamma=0.9, \
                                rar=0.98, \
                                radr=0.999, \
                                dyna=0, \
                                verbose=False)

    def addEvidence(self, hist, sv=10000):

        dates = hist.index.values

        prices = hist[['close']]
        prices = prices.ix[19:len(prices), :]

        indis = self.indicators(hist, base_N=self.base_N)

        self.no_days = indis.shape[0]

        for it in range(self.Qiter):
            s = self.discretize(indis.iloc[0], self.base_N, 1)
            lastpos = 1
            a = self.learner.querysetstate(s)
            for date in range(1, self.no_days):
                pos = a
                s = self.discretize(indis.iloc[date], self.base_N, pos)
                r = (pos - lastpos) * (prices.iloc[date] - prices.iloc[date - 1])
                lastpos = pos
                a = self.learner.query(s, r)

    def testPolicy(self, hist, sv=10000):

        dates = hist.index.values

        positions = hist[['close']]
        positions = positions.ix[19:len(positions), :]

        indis = self.indicators(hist, base_N=self.base_N)
        self.no_days = indis.shape[0]

        s = self.discretize(indis.iloc[0], self.base_N, 1)
        positions.ix[0] = 1

        for date in range(self.no_days):
            a = self.learner.querysetstate(s)
            pos = a
            positions.ix[date] = a
            s = self.discretize(indis.iloc[date], self.base_N, pos)
        positions.ix[0] = 1

        # print positions

        # for date in range(self.no_days - 1):
        #     positions.ix[date] = 200 * (positions.ix[date + 1] - positions.ix[date])

        positions = pd.DataFrame(positions, index=positions.index)
        positions = positions.set_index(positions.index.values[:])

        return positions

    def discretize(self, indi_i, base_N, pos):
        for indi in indi_i:
            pos = pos * base_N + indi
        return pos

    def indicators(self, datain, base_N):

        # dates = pd.date_range(sd, ed)
        data = datain[['close']]
        data_High = datain[['high']]
        data_Low = datain[['low']]
        data_Volume = datain[['volume']]
        data_Close = datain[['close']]
        data_Open = datain[['open']]

        def B_band_val(data, N_period=20, K_width=2):

            data = data.set_index(data.index.values[:])

            data = data / data.iloc[0]

            B_band_mean = pd.rolling_mean(data, N_period).dropna()
            B_band_std = pd.rolling_std(data, N_period).dropna()
            B_band_lower = B_band_mean - K_width * B_band_std
            B_band_upper = B_band_mean + K_width * B_band_std
            Price_value = data[N_period - 1:]
            Price_value.columns = ['Price']

            B_band_value = (data[N_period - 1:] - B_band_mean) / (2 * B_band_std)

            B_band_mean = pd.DataFrame(B_band_mean)
            B_band_mean.columns = ['B_band_mean']

            B_band_value = pd.DataFrame(B_band_value)
            B_band_value.columns = ['B_band_value']

            B_band_lower = pd.DataFrame(B_band_lower)
            B_band_lower.columns = ['B_band_lower']

            B_band_upper = pd.DataFrame(B_band_upper)
            B_band_upper.columns = ['B_band_upper']

            return B_band_value

        def MACD_val(data, N_period_short, N_period_long):

            data = data.set_index(data.index.values[:])

            macd_short = pd.ewma(data, span=N_period_short)
            macd_long = pd.ewma(data, span=N_period_long)

            macd = macd_short - macd_long
            macd_signal = pd.ewma(macd, span=14)
            macd = pd.DataFrame(macd)
            macd.columns = ['macd']
            macd_signal = pd.DataFrame(macd_signal)
            macd_signal.columns = ['signal']

            mm = macd.mean().values
            ms = macd.std().values

            return macd

        def True_range_val(data_Close, data_High, data_Low, N_period=21):

            prev_closing = data_Close[:-1]

            Range1 = np.abs((data_High.values - data_Low.values)[1:])
            Range2 = np.abs(data_High[1:].values - prev_closing.values)
            Range3 = np.abs(prev_closing.values - data_Low[1:].values)

            R = np.hstack((Range2, Range3, Range1))
            R = np.max(R, axis=1)

            Ture_r = pd.DataFrame(R)
            Ture_r = Ture_r.set_index(data_Close.index.values[1:])

            ATR = pd.ewma(Ture_r, span=N_period)
            ATR.columns = ['ATR']

            return ATR

        def OBV_val(data, data_Volume, N_period=7):

            prev_closing = data[:-1]
            dailyr = data[1:].values - prev_closing.values
            dailyup = dailyr.copy()
            dailydown = dailyr.copy() * (-1)

            indexlist1 = np.ones(dailyup.shape)
            indexlist2 = indexlist1.copy()

            indexlist1[dailyup < 0] = 0
            indexlist2[dailydown < 0] = 0

            volume_u = data_Volume[1:].values * indexlist1
            volume_d = data_Volume[1:].values * indexlist2
            volume_df = volume_u - volume_d

            volume_cd = np.cumsum(volume_df)

            volume_cd = pd.DataFrame(volume_cd)
            volume_cd = volume_cd.set_index(data.index.values[1:])

            volume_cd = pd.ewma(volume_cd, span=N_period) / 1e8
            volume_cd.columns = ['OBV']

            return volume_cd

 
        def SMA(prices, n):
            sma = pd.DataFrame(0.0, index=prices.index, columns=['SMA', 'SD'])
            for i in range(sma.shape[0]):
                sma.ix[i]['SMA'] = prices[max(0, i - n + 1):i + 1].mean()
                sma.ix[i]['SD'] = prices[max(0, i - n + 1):i + 1].std()
            return sma

        def EMA(prices, n):
            a = 1 - 2.0 / (n + 1)
            ema = pd.DataFrame(0.0, index=prices.index, columns=['EMA'])
            ntr, dmr, idx = 0, 0, 0
            for date, price in prices.iterrows():
                ntr = ntr * a + price
                dmr += a ** idx
                ema.ix[date]['EMA'] = ntr / dmr
                idx += 1
            return ema

        def RSI(prices, n1=6, n2=12, n3=24):
            U = prices - prices.shift(1)
            D = prices.shift(1) - prices
            sym = prices.columns.values[0]
            for date in prices.index:
                if not U.ix[date][sym] > 0:
                    U.ix[date] = 0.0
                if not D.ix[date][sym] > 0:
                    D.ix[date] = 0.0
            rsi = pd.DataFrame(0.0, index=prices.index, columns=['RSI1', 'RSI2', 'RSI3'])
            rsi['RSI2'] = EMA(U, n2) / (EMA(U, n2) + EMA(D, n2)) * 100
            rsi.ix[0] = [0.0, 0.0, 0.0]
            rsi = rsi.set_index(rsi.index.values[:])
            return rsi

        def MACD(prices):
            macd = pd.DataFrame(0.0, index=prices.index, columns=['DIF', 'DEM', 'OSC'])
            macd['DIF'] = EMA(prices, 12) - EMA(prices, 26)
            macd['DEM'] = EMA(pd.DataFrame(macd['DIF']), 9)
            macd['OSC'] = macd['DIF'] - macd['DEM']
            return macd

        ###

        def compute_RSI(data, n):
            delta = data.diff()
            dUp, dDown = delta.copy(), delta.copy()
            dUp[dUp < 0] = 0
            dDown[dDown > 0] = 0
            RolUp = pd.rolling_mean(dUp, n)
            RolDown = pd.rolling_mean(dDown, n).abs()
            RS = RolUp / RolDown
            RSI = 100 - 100 / (1 + RS)
            RSI = RSI.dropna()
            RSI = RSI.set_index(RSI.index.values[:])
            RSI = pd.DataFrame(RSI)
            RSI.columns = ['RSI']

            return RSI

        def Momentum(data, window=10):
            data_today = data[window - 1:].values
            data_before = data[:-(window - 1)].values
            moment = (data_today / data_before) - 1
            moment = pd.DataFrame(moment)
            moment = moment.set_index(data.index.values[window - 1:])
            moment.columns = ['momentum']

            return moment


        # print KDJ_value
        Momentum_value = Momentum(data, 10)
        RSI_value = RSI(data)
        RSI_value2 = compute_RSI(data, 10)
        # MACD_signal = MACD_val(data, 12, 26)
        MACD_signal = MACD_val(data, 20, 120)

        B_band = B_band_val(data, 20, 2)
        True_range = True_range_val(data_Close, data_High, data_Low, 21)
        OBV = OBV_val(data, data_Volume, N_period=7)
        # X = B_band.join([MACD_signal, RSI_value['RSI2'], Momentum_value]).dropna()
        X = B_band.join([OBV, True_range, MACD_signal, RSI_value2, Momentum_value]).dropna()
       

        indi = X[['B_band_value','signal','momentum','RSI']]

        cut = []
        for i in range(self.no_indi):
            cut.append([])

        self.no_days = indi.shape[0]

        for i in range(self.no_indi):
            ind_val = indi.ix[:, i].copy()
            ind_val.sort_values(inplace=True)
            cut_mark = len(ind_val) / base_N
            cut[i] = [ind_val[(cut_i) * cut_mark] for cut_i in range(1, base_N)]
            cut[i].append(ind_val[-1])
            for date in range(self.no_days):
                for cut_i in range(base_N):
                    if indi.iloc[date, i] <= cut[i][cut_i]:
                        indi.iloc[date, i] = cut_i
                        break

        return indi


StderrHandler().push_application()
StreamHandler(sys.stdout).push_application()
log = Logger('Algorithm')

from zipline.api import slippage, set_slippage
from zipline._protocol import BarData



# Slippage model to trade at the open or at a fraction of the open - close range.
class TradeAtTheOpenSlippageModel(slippage.SlippageModel):


    '''Class for slippage model to allow trading at the open
       or at a fraction of the open to close range.
    '''
    # Constructor, self and fraction of the open to close range to add (subtract)
    #   from the open to model executions more optimistically


    def __init__(self, fractionOfOpenCloseRange, spread):

        # Store the percent of open - close range to take as the execution price
        self.fractionOfOpenCloseRange = fractionOfOpenCloseRange
        self.lastPrices = None
        # Store bid/ask spread
        self.spread = spread

    def process_order(self, data, order):
        # Apply fractional slippage
        openPrice = data.current(order.sid, 'open')
        closePrice = data.current(order.sid, 'close')
        ocRange = closePrice - openPrice
        ocRange = ocRange * self.fractionOfOpenCloseRange
        targetExecutionPrice = openPrice + ocRange

        # Apply spread slippage
        targetExecutionPrice += self.spread * order.direction
        targetExecutionPrice = self.lastPrices
        log.info('\nOrder:{0} open:{1} close:{2} exec:{3} side:{4}'.format(
            order.sid, openPrice, closePrice, targetExecutionPrice, order.direction))

        # Create the transaction using the new price we've calculated.
        return (targetExecutionPrice, order.amount)


    def updateLastPrices(self,closePrice):
        '''To use the "lastClose" priceModel, you must call updateLastPrices() from your handle_data() at the END of every timestep'''
        # if self.lastPrices == None:
        #     self.lastPrices = {}
        # self.lastPrices.clear()
        # closePrice = data.current(order.sid, 'close')
        self.lastPrices = closePrice


# Setup our variables
def initialize(context):
    # SPY
    set_benchmark(sid(24))
    context.spmodel = TradeAtTheOpenSlippageModel(0.0001, .0001)

    set_slippage(context.spmodel)
    set_commission(commission.PerShare(cost=0.0, min_trade_cost=0.0))

    context.spy = sid(24)
    # context.stock1 = symbols('SPY')
    # context.BBands = deque(maxlen=50)
    # context.stocks = symbols('MMM', 'SPY', 'GOOG_L', 'PG', 'DIA')
    context.sl = StrategyLearner()
    # context.sids = {
    #     sid(16841): 0.2,
    #     sid(46631): 0.2,
    #     sid(24): 0.2,
    #     sid(14848): 0.2,
    #     sid(5061): 0.2,
    #     sid(8554): 1,
    # }

    context.iDays = 0
    context.NDays = 28  # <- Set to desired N days market is open
    context.init = 0
    context.operation = 1
    context.order_price = 0
    context.deq_flag = 0
    context.order_list = deque([], maxlen=15)

    # Algorithm will only take long positions.
    # It will stop if encounters a short position.
    # set_long_only()


    schedule_function(model_init, date_rules.every_day(), time_rules.market_close(), half_days=True)
    # schedule_function(model, date_rules.every_day(), time_rules.market_open(), half_days=True)
    schedule_function(find_op, date_rules.every_day(), time_rules.market_close(), half_days=True)
    schedule_function(rebalance, date_rules.every_day(), time_rules.market_close(), half_days=True)


# Rebalance daily.
def model_init(context, data):
    # Track our position
    context.iDays += 1

    if context.init == 0:
        hist = data.history(context.spy, ['open', 'high', 'volume', 'low', 'close'], 1300, '1d')
        log.info(hist.iloc[:])
        context.sl.addEvidence(hist.iloc[:], sv=10000)
        context.init = 1
        log.info("model init!")

    else:
        if (context.iDays % context.NDays) == 1:
            current_position = context.portfolio.positions[context.spy].amount
            record(position_size=current_position)

            # Load historical data for the stocks
            hist = data.history(context.spy, ['open', 'high', 'volume', 'low', 'close'], 28 + 19, '1d')
            context.sl.addEvidence(hist, sv=10000)

# Rebalance daily.
def find_op(context, data):
    # Track our position


    hist = data.history(context.spy, ['open', 'high', 'volume', 'low', 'close'], 22, '1d')
    log.info(hist.iloc[-1])

    orders = context.sl.testPolicy(hist, sv=10000)
    log.info(orders.iloc[-1])
    dates = orders.index.values

    if orders.get_value(dates[-1], 'close') == 2:
        context.operation =2
        context.order_price = hist.iloc[-1]['close']
        # order_target_percent(context.spy, context.sids[sid(8554)])
    elif orders.get_value(dates[-1], 'close') == 0:
        context.operation = 0
        context.order_price = hist.iloc[-1]['close']
    else:
        context.operation = 1
        context.order_price = hist.iloc[-1]['close']

    context.spmodel.updateLastPrices(context.order_price)


# Rebalance daily.
def rebalance(context, data):
    # Track our position
    current_position = context.portfolio.positions[context.spy].amount

    record(stock_p=data.current(context.spy, "price"),
           position_size=current_position)

    if context.operation == 2:
        order_id=order_target(context.spy, 50)#, limit_price= context.order_price+0.15, stop_price  = context.order_price-0.15)
        order  = get_order(order_id)
        log.info("long")
        log.info(order)
        # order_target_percent(context.spy, context.sids[sid(8554)])
    elif context.operation == 0:
        order_id=order_target(context.spy, -50)#, limit_price=context.order_price + 0.15, stop_price=context.order_price - 0.15)
        order  = get_order(order_id)
        log.info("short")
        log.info(order)
    else:
        order_id=order_target(context.spy, 0)#, limit_price=context.order_price + 0.15, stop_price=context.order_price - 0.15)
        order  = get_order(order_id)
        log.info("empty")
        log.info(order)


    context.order_list.append(order_id)
    log.info("order_list")
    log.info(context.order_list)
    context.deq_flag = context.deq_flag +1

    if context.deq_flag == 2:
        order_id = context.order_list.popleft()
        cancel_order(order_id)
        order = get_order(order_id)
        log.info("check_cancel_order")
        log.info(order)
        if order != None and order['filled'] !=0:
            log.info('Actual Cost basis: ' + str(context.portfolio.positions[context.spy].cost_basis-order['commission']/order['filled']))
            log.info('Cost basis: ' + str(
                context.portfolio.positions[context.spy].cost_basis))
            log.info('Commission: ' + str(order['commission'] / order['filled']))

            log.info('************end_of_day**************')
        context.deq_flag = context.deq_flag -1


def _test_args():
    """Extra arguments to use when zipline's automated tests run this example.
    """
    import pandas as pd

    return {
        'start': pd.Timestamp('2016-01-01', tz='utc'),
        'end': pd.Timestamp('2017-05-01', tz='utc'),
    }

/ Explicitly sort by timestamp and sequence for a deterministic last observation.
trades:([]seq:1 2 3 4 5 6;sym:`AA`BB`AA`CC`AA`BB;ts:59 58 60 61 60 59;px:100 200 102 300 103 201;qty:2 3 4 1 5 2)
quotes:`sym`ts`seq xasc ([]seq:1 2 3 4 5;sym:`AA`BB`AA`AA`BB;ts:58 59 60 60 62;bid:98 198 99 101 199)
lastpx:{exec px from `sym xasc 0!select last px by sym from `ts`seq xasc trades}
minutes:{flip value flip 0!select total:sum qty by minute:ts div 60 from trades}
bids:{exec -1^bid from aj[`sym`ts;trades;quotes]}
unmatched:{sum null exec bid from aj[`sym`ts;trades;quotes]}
bars:{flip value flip value select o:first px,h:max px,l:min px,c:last px by minute:ts div 60 from `ts`seq xasc trades}
bbvwap:{exec qty wavg px from trades where sym=`BB}
rollingqty:{exec vol from `seq xasc update vol:2 msum qty by sym from `ts`seq xasc trades}

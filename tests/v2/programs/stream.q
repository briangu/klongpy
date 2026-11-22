/ Reference implementation for the same in-memory workload.
init:{[s;qs;qt;qb] trades::([]sym:s;ts:(count s)#-1j;px:(count s)#0j);quotes::([]sym:qs;ts:qt;bid:qb);1}
ingest:{[s;t;p] `trades insert (s;t;p);1}
latest:{exec px from `sym xasc 0!select last px by sym from trades}
asofbid:{exec -1^bid from aj[`sym`ts;`sym xasc 0!select last ts by sym from trades;quotes]}
nrows:{count trades}

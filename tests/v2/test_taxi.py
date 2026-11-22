"""Run the NYC example's four reports with hand-computed fares and counts."""
import json

from .runtime import ROOT, evaluate


def test_taxi_rollup_and_result_shapes(host, tmp_path):
    helper = (ROOT / "examples/db/taxi-queries.kg").as_posix().replace('"', '""')
    source = (
        f'.l("{helper}");d::.db(:{{}});'
        'd("CREATE TABLE trips(cab_type VARCHAR, passenger_count INTEGER, y INTEGER, '
        'trip_distance DOUBLE, fare DOUBLE)");'
        'd("INSERT INTO trips VALUES (\'green\',1,2014,0.5,3),(\'green\',1,2014,0.5,9),'
        '(\'yellow\',1,2014,1.0,24),(\'yellow\',1,2014,-0.5,NULL),'
        '(\'yellow\',NULL,NULL,NULL,NULL)");tripReports(d);'
        'd("DELETE FROM trips WHERE passenger_count IS NOT NULL");tripReports(d);'
        'd("DELETE FROM trips");tripReports(d);1'
    )
    output = evaluate(host, source, tmp_path).splitlines()
    assert output[-1] == "1"
    reports = [json.loads(line) for line in output[:-1]]
    assert reports == [
        [["green", 2], ["yellow", 3]], [[1, 12], [None, None]],
        [[1, 2014, 4], [None, None, 1]],
        [[1, 2014, 1, 3], [1, 2014, -1, 1], [None, None, None, 1]],
        [["yellow", 1]], [[None, None]], [[None, None, 1]], [[None, None, None, 1]],
        [], [], [], [],
    ]

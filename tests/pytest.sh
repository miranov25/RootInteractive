# source $RootInteractive/tests/pytest.sh
# To understand commend read  jq totorial https://github.com/rjz/jq-tutorial and jq blog https://earthly.dev/blog/jq-select/
#


pytestExample(){
  time pytest --workers 6  --tests-per-worker 20 test_bokehDrawSA.py --json-report --json-report-file test_bokehDrawSA.json
}

pytestCompress(){
   time pytest --workers 6  --tests-per-worker 20 test_Compression.py --json-report --json-report-file test_Compression.json

   # {nodeid,call}|del(.call.stdout)
   #jq -r '{environment,tests}' test_Compression.json
   jq -r '{environment,tests}' test_Compression.json
  # print environment and tests
  jq -r '{environment,tests}|del(.tests[].call.stdout)' test_Compression.json
  #
  jq -r '{environment,tests}|del(.tests[].call.stdout,.tests[].keywords,.tests[].teardown,.tests[].setup)' test_Compression.json
  # making  csv file
  #https://jqterm.com/7e2ce231a4bfcbb4a1025e3c04f97ae5?query=%5B%7Benvironment%3A.environment.Platform%2C%20%22id%22%3A.tests%5B%5D.nodeid%2C%20%22tests%22%3A%20.tests%5B%5D.call%7D%5D%7C.%5B%5D%7C%5B.environment%2C.id%2C.tests.duration%5D%7C%40csv%0A
   jq -r '[{environment:.environment.Platform, "id":.tests[].nodeid, "tests": .tests[].call}]|.[]|[.environment,.id,.tests.duration]' test_Compression.json
  jq -r '[{environment:.environment.Platform, "id":.tests[].nodeid, "tests": .tests[].call}]|.[]|[.environment,.id,.tests.duration]|@csv' test_Compression.json
  jq -r '[{environment:.environment.Platform, "id":.tests[].nodeid, "tests": .tests[].call}]|.[]|[.environment,.id,.tests.duration]|@csv' test_Compression.json > test_Compression.csv
}

void makeTestFilter(){
  #metadata can be added per  ... directory
  metadata=$(echo --metadata hostname "$HOSTNAME")
  metadata+=$(echo " "--metadata username "$USER")
  #metadata+=$(echo " "--metadata cpuName \"$(cat /proc/cpuinfo | grep 'vendor_id' | uniq)\")
  metadata+=$(echo " "  --metadata nCPU "$(cat /proc/cpuinfo | grep processor | wc -l)")
  time pytest $metadata  --workers 6  --tests-per-worker 20 test_*.py --json-report --json-report-file testAll.json


}
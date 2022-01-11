# source $RootInteractive/tests/pytest.sh
#
#  To understand jq command  read  jq totorial https://github.com/rjz/jq-tutorial and jq blog https://earthly.dev/blog/jq-select/
#  Issue descruption in https://github.com/miranov25/RootInteractive/issues/177

export pathJQ='[paths | map(.|tostring) | join(".")| gsub("(.\\d+)"; "[]")]|unique'
export tostringJQ='map(.|tostring) |join(":")'

alias pathJQ='jq -r "$pathJQ" '

init(){
  cat <<HELP_USAGE | cat
  init
  pathJQ          -> return expanded list of the keys
  makePyTest      -> make pytest store results and metadata  in json files
  uploadPyTest    -> upload test on the server
HELP_USAGE
}

pytestExampleQueries(){

  jq '.tests[]|select(.call.duration >10)|[.nodeid,.call.duration]|map(.|tostring) |join(":")' ./v0-01-03-rev1-26-gae95bc0/InteractiveDrawing/bokeh/testGS*.json
  jq '.|select(.tests[].call.duration >10)|[.tests[].nodeid,.tests[].call.duration]|map(.|tostring) |join(":")' ./v0-01-03-rev1-26-gae95bc0/InteractiveDrawing/bokeh/testGS*.json
  # select testNames, testnode iD and print results in table
  jq '.|{tName:.environment.testName,t:.tests[]}|[.tName,.t.call.duration,.t.nodeid]|map(.|tostring) |join("  ")' ./v0-01-03-rev1-26-gae95bc0/InteractiveDrawing/bokeh/testGSI*.json
  # the same with
  jq '.|{tName:.environment.testName,t:.tests[]}|select(.t.call.duration >10)|[.tName,.t.call.duration,.t.nodeid]|map(.|tostring) |join("  ")' ./v0-01-03-rev1-26-gae95bc0/InteractiveDrawing/bokeh/testGSI*.json

}

pytestExampleJq(){
   cat <<HELP_USAGE | cat
    makeTest and store data
    Parameters:
      testWorkers  - number of workers to test
HELP_USAGE
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


makePyTest(){
 cat <<HELP_USAGE | cat
    makePyTest
    Parameters:
      1.) param $1 - testWorkers  - number of workers to test
      2.) param $2 - testName
      3.) param $3 - testPrefixServer
      4.) param $4 - testServer
    Example usage:
        makePyTest 6 test6 /lustre/alice/users/miranov/tests/RootInteractive alice
HELP_USAGE
    [[ $# -ne 4 ]] &&return
  testWorkers=${1:-5}
  testName=${2:-$(pwd | xargs basename)}
  testPrefixServer=${3}
  testServer=${4}
  testServerPrefix
  echo makePyTest $testWorkers $testName ${testPrefixServer} ${testServer}
  #return
  #metadata can be added per  ... directory
  metadata=$(echo --metadata hostname "$HOSTNAME ")
  metadata+=$(echo --metadata testName "$testName ")
  metadata+=$(echo " "--metadata username "$USER ")
  #metadata+=$(echo " "--metadata cpuName \"$(cat /proc/cpuinfo | grep 'vendor_id' | uniq)\")
  metadata+=$(echo " "  --metadata nCPU "$(cat /proc/cpuinfo | grep processor | wc -l) ")
  metadata+=$(echo --metadata testWorkers "$testWorkers ")
  metadata+=$(echo --metadata gitDescribe "$(git describe --tags) ")
  echo metadata $metadata
  echo pytest $metadata  --workers $testWorkers  --tests-per-worker 20 test_*.py --json-report --json-report-file $testName.json
  time pytest $metadata  --workers $testWorkers  --tests-per-worker 20 test_*.py --json-report --json-report-file $testName.json
  uploadPyTest $testName "${testPrefixServer}" "${testServer}"
}

uploadPyTest(){
     cat <<HELP_USAGE | cat
    makeTest and store data
    Test stored in the server param 3 ${3} deiterctory prefix ${2}
    wdir = {pwd| sed s_.*/RootInteractive__
    Parameters:
      1 param   - $1 - testName
      2 prefix  - $2 - prefix
      3 server   - $3 - alice
     uploadPyTest test6 /lustre/alice/users/miranov/tests/RootInteractive alice:
HELP_USAGE
    [[ $# -ne 3 ]] &&return
    testName=$1
    prefix=$2
    server=$3
    prefix+="/$(date +'%Y')/$(git describe --tags)"
    dirPath0=$(pwd| sed s_.*/RootInteractive__)
    mkdir -p ${prefix}/$dirPath0/
    ssh -X $server mkdir -p  /lustre/${prefix}/$dirPath0/
    rsync -avzt  $testName.json $server${prefix}/$dirPath0/$testName.json
}


init;
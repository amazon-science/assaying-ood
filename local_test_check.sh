#!/bin/bash
### Runs static code checks and unit tests.

# To be safe, switch into the folder that contains this script.
cd "$( cd "$( dirname "$0" )" && pwd )" || exit


# The following dirs are included in the check by default
# [note: exclude dirs may be read from pyproject.toml/setup.cfg and may override this]
INC_DIRS="bin src test test_integration"


install_hook() {
    # install pre-commit hook
    ln -fs "$(pwd)/.githooks/pre-commit" "$(pwd)/.git/hooks/pre-commit"
}

call_test() {
    # add -s option if you want to see output of print statements
    env PYTHONPATH=./src python -m pytest -c /dev/null -v --cov-report term-missing "${1}" "$@"
}

run_tests() {
    call_test test "$@"
}

run_manual_tests() {
    call_test "$@"
}

run_integration_tests() {
    call_test test_integration "$@"
}


run_format() {
    black $INC_DIRS
    isort $INC_DIRS
    flake8 $INC_DIRS
}


check_format () {
    black $INC_DIRS --check && isort $INC_DIRS --check && flake8 $INC_DIRS
    CODE_CHECK=$?

    RED='\033[1;31m'
    NC='\033[0m' # No Color
    if [[ $CODE_CHECK -ne 0 ]]
    then
	echo -e "${RED}*** !!! Your code is not well formatted! Please run './local_test_check.sh --format' to auto-format !!! ***${NC}"
    else
	echo "Formatter and linter checks passed."
    fi
}


print_help() {
   echo "Syntax: local_test_check.sh [options]"
   echo "options:"
   echo "--format (--f)     Run required formatters."
   echo "--test (--t)       Only run tests."
   echo "--inttest (--it)   Run integration tests."
   echo "--mantest (--mt)   Run manual tests."
#Example: ./local_test_check.sh --mt test_integration -k 'test_cli_adaptation_on_s3_dataset'
   echo "--install (--i)    Install git hooks."
   echo "--check (--c)      Only run format checks."
   echo
}
die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error


# Parse options
while getopts ab:c:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    c | check )    check_format ;;
    f | format )   run_format ;;
    t | test )     run_tests ;;
    it | inttest ) run_integration_tests ;;
    mt | mantest ) run_manual_tests ${@:2} ;;
    i | install )  install_hook ;;
    h | help )     print_help ;;
    ??* )          die "Illegal option --$OPT" ;;  # bad long option
    ? )            exit 2 ;;  # bad short option (error reported via getopts)
  esac
done
# default
if [ $OPTIND -eq 1 ]; then
    print_help
    echo
    echo "No option provided, running default checks..."
    run_tests
    check_format
fi
shift $((OPTIND-1)) # remove parsed options and args from $@ list

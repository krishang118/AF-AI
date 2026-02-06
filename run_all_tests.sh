#!/bin/bash
# Comprehensive Test Suite Runner

echo "================================================================================"
echo "COMPREHENSIVE TEST SUITE - FINAL AUDIT"
echo "================================================================================"
echo ""

cd /Users/krishangsharma/Downloads/BNC
source venv/bin/activate

total_passed=0
total_failed=0

run_test() {
    test_name=$1
    test_file=$2
    
    echo "--------------------------------------------------------------------------------"
    echo "TEST: $test_name"
    echo "--------------------------------------------------------------------------------"
    
    python "$test_file" 2>&1 | grep -v FutureWarning | grep -v ChainedAssignment > /tmp/test_output.txt
    
    if grep -q "PASS\|SUCCESS\|passed" /tmp/test_output.txt && ! grep -q "FAIL" /tmp/test_output.txt; then
        echo "✅ PASS: $test_name"
        ((total_passed++))
    else
        echo "❌ FAIL: $test_name"
        ((total_failed++))
        echo "Output:"
        cat /tmp/test_output.txt | tail -20
    fi
    echo ""
}

# Run all tests
run_test "Period Edge Cases (13 tests)" "test_comprehensive_periods.py"
run_test "All Period Types (7 tests)" "test_all_period_types.py"
run_test "Quarter Formats (5 tests)" "test_quarter_formats.py"
run_test "Month Interception (5 tests)" "test_month_interception.py"
run_test "Production Readiness (12 tests)" "audit_production_readiness.py"
run_test "Annual Rate - Weekly" "test_annual_rate_weekly.py"

echo "================================================================================"
echo "FINAL SUMMARY"
echo "================================================================================"
echo "Total Passed: $total_passed"
echo "Total Failed: $total_failed"
echo ""

if [ $total_failed -eq 0 ]; then
    echo "🎉 ALL TEST SUITES PASSED!"
    exit 0
else
    echo "❌ $total_failed TEST SUITE(S) FAILED"
    exit 1
fi

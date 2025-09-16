#!/usr/bin/env python3
"""
Test script for MultiPL-E API Server
Tests all supported languages with sample programs
"""

import requests
import json
import time
import os
from typing import Dict, Any

# API Configuration - configurable for remote/SLURM nodes
API_HOST = os.getenv('API_HOST', '147.46.15.142')
API_PORT = int(os.getenv('API_PORT', '8888'))
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

# Test programs for each language
TEST_PROGRAMS = {
    "r": {
        "program": """
# Simple R function
add_numbers <- function(a, b) {
    return(a + b)
}

# Test the function
result <- add_numbers(5, 3)
print(paste("Result:", result))

# Test case
stopifnot(add_numbers(5, 3) == 8)
""",
        "expected_status": "OK"
    },

    "julia": {
        "program": """
# Simple Julia function
function add_numbers(a, b)
    return a + b
end

# Test the function
result = add_numbers(5, 3)
println("Result: ", result)

# Test case
@assert add_numbers(5, 3) == 8
""",
        "expected_status": "OK"
    },

    "lua": {
        "program": """
-- Simple Lua function
function add_numbers(a, b)
    return a + b
end

-- Test the function
result = add_numbers(5, 3)
print("Result: " .. result)

-- Test case
assert(add_numbers(5, 3) == 8)
""",
        "expected_status": "OK"
    },

    "racket": {
        "program": """
#lang racket

; Simple Racket function
(define (add-numbers a b)
  (+ a b))

; Test the function
(define result (add-numbers 5 3))
(printf "Result: ~a~n" result)

; Test case
(unless (= (add-numbers 5 3) 8)
  (error "Test failed"))
""",
        "expected_status": "OK"
    },

    "ml": {
        "program": """
(* Simple OCaml function *)
let add_numbers a b = a + b;;

(* Test the function *)
let result = add_numbers 5 3;;
Printf.printf "Result: %d\\n" result;;

(* Test case *)
assert (add_numbers 5 3 = 8);;
""",
        "expected_status": "OK"
    }
}

def test_health_check() -> bool:
    """Test health check endpoint"""
    try:
        print(f"Testing connection to {API_BASE_URL}/health...")
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed: {data['status']}")
            print(f"  Supported languages: {data['supported_languages']}")
            return True
        else:
            print(f"✗ Health check failed: HTTP {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except requests.exceptions.ConnectionError as e:
        print(f"✗ Connection failed: Cannot connect to {API_BASE_URL}")
        print(f"  Error: {e}")
        print(f"  Check if:")
        print(f"    1. Server is running on {API_HOST}:{API_PORT}")
        print(f"    2. Host IP address is correct")
        print(f"    3. Port is not blocked by firewall")
        return False
    except requests.exceptions.Timeout as e:
        print(f"✗ Health check timeout: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_languages_endpoint() -> bool:
    """Test languages endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/languages", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Languages endpoint passed")
            print(f"  Languages: {data['supported_languages']}")
            return True
        else:
            print(f"✗ Languages endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Languages endpoint failed: {e}")
        return False

def test_evaluate_language(language: str, test_data: Dict[str, Any]) -> bool:
    """Test evaluation for a specific language"""
    try:
        payload = {
            "program": test_data["program"],
            "language": language
        }

        print(f"\\nTesting {language}...")
        start_time = time.time()

        response = requests.post(
            f"{API_BASE_URL}/evaluate",
            json=payload,
            timeout=30  # Longer timeout for code execution
        )

        execution_time = (time.time() - start_time) * 1000

        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "Unknown")

            print(f"  Status: {status}")
            print(f"  Execution time: {execution_time:.1f}ms (client)")
            print(f"  Server execution time: {data.get('execution_time_ms', 'N/A')}ms")

            if data.get("stdout"):
                print(f"  Stdout: {data['stdout'].strip()}")
            if data.get("stderr"):
                print(f"  Stderr: {data['stderr'].strip()}")

            if status == test_data["expected_status"]:
                print(f"✓ {language} test passed")
                return True
            else:
                print(f"✗ {language} test failed: expected {test_data['expected_status']}, got {status}")
                return False
        else:
            print(f"✗ {language} test failed: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"  Error: {error_data.get('error', 'Unknown error')}")
            except:
                pass
            return False

    except requests.exceptions.RequestException as e:
        print(f"✗ {language} test failed: {e}")
        return False

def test_error_cases() -> bool:
    """Test error handling"""
    print(f"\\nTesting error cases...")

    # Test missing data
    try:
        response = requests.post(f"{API_BASE_URL}/evaluate", json={}, timeout=5)
        if response.status_code == 422:  # FastAPI validation error
            print("✓ Missing data error handled correctly")
        else:
            print(f"✗ Missing data error not handled: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Missing data test failed: {e}")
        return False

    # Test unsupported language
    try:
        response = requests.post(
            f"{API_BASE_URL}/evaluate",
            json={"program": "print('hello')", "language": "unsupported"},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "UnsupportedLanguage":
                print("✓ Unsupported language handled correctly")
            else:
                print(f"✗ Unsupported language not handled: {data.get('status')}")
                return False
        else:
            print(f"✗ Unsupported language test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Unsupported language test failed: {e}")
        return False

    return True

def main():
    """Run all tests"""
    print("=== MultiPL-E API Server Test Suite ===")
    print(f"Testing API at: {API_BASE_URL}")
    print(f"Host: {API_HOST}")
    print(f"Port: {API_PORT}")
    print()

    # Test basic endpoints
    if not test_health_check():
        print("\\n❌ Health check failed.")
        print("\\n🔧 Troubleshooting:")
        print(f"1. Check if server is running: ssh {API_HOST} 'ps aux | grep api_server'")
        print(f"2. Test port accessibility: telnet {API_HOST} {API_PORT}")
        print(f"3. Check firewall: ssh {API_HOST} 'iptables -L | grep {API_PORT}'")
        print(f"4. Override IP/port: export API_HOST=<ip> API_PORT=<port>")
        return False

    if not test_languages_endpoint():
        print("\\nLanguages endpoint failed.")
        return False

    # Test each language
    passed = 0
    total = len(TEST_PROGRAMS)

    for language, test_data in TEST_PROGRAMS.items():
        if test_evaluate_language(language, test_data):
            passed += 1

    # Test error cases
    error_tests_passed = test_error_cases()

    # Results
    print(f"\\n=== Test Results ===")
    print(f"Language tests: {passed}/{total} passed")
    print(f"Error handling tests: {'✓' if error_tests_passed else '✗'}")

    if passed == total and error_tests_passed:
        print("\\n🎉 All tests passed!")
        return True
    else:
        print(f"\\n❌ Some tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
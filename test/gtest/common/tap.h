/*
 * The MIT License
 *
 * Copyright (c) 2011 Bruno P. Kinoshita <http://www.kinoshita.eti.br>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * @author Bruno P. Kinoshita <http://www.kinoshita.eti.br>
 * @since 0.1
 */

#ifndef TAP_H_
#define TAP_H_

#include <list>
#include <iostream>
#include <map>
#include <fstream>
#include <string>
#include <algorithm>

namespace tap {

#ifdef GTEST_TAP_13_DIAGNOSTIC
// based on http://stackoverflow.com/a/7724536/831180
static std::string replace_all_copy(
  std::string const& original,
  std::string const& before,
  std::string const& after
) {
  using namespace std;

  if (before == after) return string(original);

  string retval;
  if (before.length() == after.length()) retval.reserve(original.size());

  basic_string <char>::const_iterator end = original.end();
  basic_string <char>::const_iterator current = original.begin();
  basic_string <char>::const_iterator next =
    search(current, end, before.begin(), before.end());

  while ( next != end ) {
    retval.append( current, next );
    retval.append( after );
    current = next + before.size();
    next = search(current, end, before.begin(), before.end());
  }
  retval.append( current, next );
  return retval;
}
#endif

class TestResult {

 private:
  int number;
  std::string status;
  std::string name;
  std::string comment;
  bool skip;

 public:
  std::string getComment() const {
    std::stringstream ss;
    if (this->skip) {
      ss << "# SKIP " << this->comment;
    } else if (!this->comment.empty()) {
      ss << "# " << this->comment;
    }
    return ss.str();
  }

  const std::string& getName() const {
    return name;
  }

  int getNumber() const {
    return number;
  }

  const std::string& getStatus() const {
    return status;
  }

  bool getSkip() const {
    return skip;
  }

  void setComment(const std::string& comment) {
    this->comment = comment;
  }

  void setName(const std::string& name) {
    this->name = name;
  }

  void setNumber(int number) {
    this->number = number;
  }

  void setStatus(const std::string& status) {
    this->status = status;
  }

  void setSkip(bool skip) {
    this->skip = skip;
  }

  std::string toString() const {
    std::stringstream ss;
    ss << this->status << " " << this->number << " " << this->name;
#ifdef GTEST_TAP_13_DIAGNOSTIC
    std::string comment_text = this->getComment();
    if (!comment_text.empty()) {
      ss << std::endl
       << "# Diagnostic" << std::endl
       << "  ---" << std::endl
       << "  " << replace_all_copy(this->getComment(), "\n", "\n  ");
    }
#endif
    return ss.str();
  }
};

class TestSet {

 private:
  std::list<TestResult> testResults;

 public:
  const std::list<TestResult>& getTestResults() const {
    return testResults;
  }

  void addTestResult(TestResult& testResult) {
    testResult.setNumber((this->getNumberOfTests() + 1));
    this->testResults.push_back(testResult);
  }

  int getNumberOfTests() const {
    return this->testResults.size();
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "1.." << this->getNumberOfTests() << std::endl;
    for (std::list<TestResult>::const_iterator ci = this->testResults.begin();
	 ci != this->testResults.end(); ++ci) {
      TestResult testResult = *ci;
      ss << testResult.toString() << std::endl;
    }
    return ss.str();
  }
};

class TapListener: public ::testing::EmptyTestEventListener {

 private:
  std::map<std::string, tap::TestSet> testCaseTestResultMap;

  void addTapTestResult(const testing::TestInfo& testInfo) {
    tap::TestResult tapResult;
    tapResult.setName(testInfo.name());
    tapResult.setSkip(!testInfo.should_run());

    const testing::TestResult *testResult = testInfo.result();
    int number = testResult->total_part_count();
    tapResult.setNumber(number-1);
    if (testResult->HasFatalFailure()) {
      tapResult.setStatus("Bail out!");
    } else if (testResult->Failed()) {
      tapResult.setStatus("not ok");
      tapResult.setComment(testResult->GetTestPartResult(number-1).summary());
    } else {
      tapResult.setStatus("ok");
    }

    this->addNewOrUpdate(testInfo.test_case_name(), tapResult);
  }

  std::string getCommentOrDirective(const std::string& comment, bool skip) {
    std::stringstream commentText;

    if (skip) {
      commentText << " # SKIP " << comment;
    } else if (!comment.empty()) {
      commentText << " # " << comment;
    }

    return commentText.str();
  }

  void addNewOrUpdate(const std::string& testCaseName, tap::TestResult testResult) {
    std::map<std::string, tap::TestSet>::const_iterator ci =
      this->testCaseTestResultMap.find(testCaseName);
    if (ci != this->testCaseTestResultMap.end()) {
      tap::TestSet testSet = ci->second;
      testSet.addTestResult(testResult);
      this->testCaseTestResultMap[testCaseName] = testSet;
    } else {
      tap::TestSet testSet;
      testSet.addTestResult(testResult);
      this->testCaseTestResultMap[testCaseName] = testSet;
    }
  }

public:
  virtual void OnTestEnd(const testing::TestInfo& testInfo) {
    //printf("%s %d - %s\n", testInfo.result()->Passed() ? "ok" : "not ok", this->testNumber, testInfo.name());
    this->addTapTestResult(testInfo);
  }

  virtual void OnTestProgramEnd(const testing::UnitTest& unit_test) {
    //--- Write the count and the word.
    std::map<std::string, tap::TestSet>::const_iterator ci;
    for (ci = this->testCaseTestResultMap.begin();
	 ci != this->testCaseTestResultMap.end(); ++ci) {
      const tap::TestSet& testSet = ci->second;
#ifdef GTEST_TAP_PRINT_TO_STDOUT
      std::cout << "TAP version 13" << std::endl;
      std::cout << testSet.toString();
#else
      std::string ext = ".tap";
      std::ofstream tapFile;
      tapFile.open((ci->first + ext).c_str());
      tapFile << testSet.toString();
      tapFile.close();
#endif
    }
  }
};

} // namespace tap

#endif // TAP_H_

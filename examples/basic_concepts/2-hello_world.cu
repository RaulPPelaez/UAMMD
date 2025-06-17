/* Raul P. Pelaez 2021
   Hello world with UAMMD
   We will start from the previous example and see the first utility provided by
   System, the log function. System::log allows to, well, log messages with
   several levels of priority. Internally UAMMD modules use this utility to
   print everything ranging from informational messages to critical errors and
   low level debug messages. In this example we will use System::log to print a
   simple message along with the date.
 */

// uammd.cuh is the basic uammd include containing, among other things, the
// System struct.
#include <ctime> //For time and ctime
#include <uammd.cuh>
using namespace uammd;
// The main function will initialize an UAMMD environment, print a message, then
// destroy it and exit
int main(int argc, char *argv[]) {
  // Initialize System
  auto sys = std::make_shared<System>(argc, argv);
  // Unles something goes wrong System creation logs messages using level
  // MESSAGE, which prints low priority information to stderr. There are a lot
  // more levels to choose from, each associated with a number. From highest to
  // lowest priority these are: CRITICAL=0, ERROR, EXCEPTION, WARNING, MESSAGE,
  // STDERR, STDOUT,DEBUG, DEBUG1, DEBUG2, DEBUG3, DEBUG4, DEBUG5, DEBUG6,
  // DEBUG7=14 For example, MESSAGE is associated with log level number 5. Lets
  // print something using the MESSAGE level:
  sys->log<System::MESSAGE>("Hello from UAMMD");
  // Lets also print todays date, this time as a WARNING:
  auto currentTime = time(nullptr);
  // Notice that System::log works as C's printf, with a format string and then
  // arguments.
  sys->log<System::WARNING>("Current time is: %s", ctime(&currentTime));
  // The maximum log level printed can be controled through the MAXLOGLEVEL
  // compile macro (which can be selected in the Makefile). The default is 5,
  // which will print up to MESSAGE. The special level CRITICAL will terminate
  // the execution of the program with an error code. Unless the log level
  // STDOUT is used, all messages will be issued to stderr. Since it is known at
  // compile time, any log calls with levels above the maximum one will not be
  // compiled and thus will incur
  //  no performance penalty
  // Destroy the UAMMD environment and exit
  sys->finish();
  return 0;
}

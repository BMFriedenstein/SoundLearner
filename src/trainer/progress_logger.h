/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * progress_logger.h
 *
 *  Created on: 14 Jan 2019
 *      Author: brandon
 */

#ifndef SRC_TRAINER_PROGRESS_LOGGER_H_
#define SRC_TRAINER_PROGRESS_LOGGER_H_

#include <string>

namespace logging{

class ProgressLogC{
 private:
  std::string filename_;
  bool cpy_to_console_;
 public:
  ProgressLogC(std::string filename,bool to_console):
    filename_(filename),
    cpy_to_console_(to_console) { }
  void Start();
  void WriteLine(const std::string& line);

  inline void operator<<(const std::string& line){
    WriteLine(line);
  }
  static void WriteFile(const std::string filename,
                        const std::string& content);
};
}

#endif  // SRC_TRAINER_PROGRESS_LOGGER_H_

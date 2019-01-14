/*
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * progress_logger.cpp
 *
 *  Created on: 14 Jan 2019
 *      Author: brandon
 */


#include "progress_logger.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

namespace logging {
void ProgressLogC::Start(){
  std::fstream file(filename_, std::fstream::out);
  file << "";
  file.close();
}

void ProgressLogC::WriteLine(const std::string& line) {
  if(cpy_to_console_){
    std::cout << line << std::endl;
  }
  std::ofstream out_stream(filename_, std::ios::app);
  out_stream << line + "\n";
  out_stream.close();
}

void ProgressLogC::WriteFile(const std::string filename,
                             const std::string& content) {
  std::cout << filename << std::endl;
  std::fstream file(filename, std::fstream::out);
  file << content+ "\n";
  file.close();
}
}

#pragma once

#include "reagent/serving/core/DecisionService.h"
#include "reagent/serving/core/Headers.h"

#include "SimpleWebServer/client_http.hpp"
#include "SimpleWebServer/server_http.hpp"
using HttpServer = SimpleWeb::Server<SimpleWeb::HTTP>;
using HttpClient = SimpleWeb::Client<SimpleWeb::HTTP>;

namespace reagent {
class Server {
 public:
  Server(std::shared_ptr<DecisionService> decisionService, int port);

  void start();
  void shutdown();

 protected:
  HttpServer server_;
  std::shared_ptr<std::thread> serverThread_;
  std::shared_ptr<DecisionService> decisionService_;
  int port_;
};
}  // namespace reagent

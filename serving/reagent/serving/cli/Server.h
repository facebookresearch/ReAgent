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
  Server(std::shared_ptr<DecisionService> _decisionService);

  void shutdown();

 protected:
  HttpServer server;
  std::shared_ptr<std::thread> serverThread;
  std::shared_ptr<DecisionService> decisionService;
};
}  // namespace reagent

#include "reagent/serving/cli/Server.h"

namespace reagent {
Server::Server(std::shared_ptr<DecisionService> _decisionService)
    : decisionService(_decisionService) {
  server.config.port = 3000;

  server.resource["^/api/TODO$"]["POST"] =
      [this](std::shared_ptr<HttpServer::Response> response,
             std::shared_ptr<HttpServer::Request> request) {
        auto content = json::parse(request->content.string());

        json retval = {{"status", "OK"}};
        response->write(SimpleWeb::StatusCode::success_ok, retval.dump(2));
      };

  serverThread.reset(new std::thread([this]() { server.start(); }));
}

void Server::shutdown() {
  server.stop();
  if (serverThread->joinable()) {
    serverThread->join();
  }
}

}  // namespace reagent
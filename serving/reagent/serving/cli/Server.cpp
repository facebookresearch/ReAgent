#include "reagent/serving/cli/Server.h"

namespace reagent {
Server::Server(std::shared_ptr<DecisionService> decisionService, int port)
    : decisionService_(decisionService), port_(port) {}

void Server::start() {
  server_.config.port = port;

  server_.resource["^/api/request$"]["POST"] =
      [this](std::shared_ptr<HttpServer::Response> response,
             std::shared_ptr<HttpServer::Request> request) {
        try {
          VLOG(1) << "REQUEST";
          auto content = json::parse(request->content.string());
          VLOG(1) << "Got request: " << content;
          DecisionRequest decisionRequest = content;
          auto decisionResponse =
              decisionService_->attachIdAndProcess(decisionRequest);
          json responseJson = decisionResponse;

          response->write(SimpleWeb::StatusCode::success_ok,
                          responseJson.dump(2));
        } catch (const std::exception& e) {
          LOG(ERROR) << "GOT ERROR: " << e.what();
          response->write(SimpleWeb::StatusCode::client_error_bad_request,
                          e.what());
        }
      };

  server.resource["^/api/feedback$"]["POST"] =
      [this](std::shared_ptr<HttpServer::Response> response,
             std::shared_ptr<HttpServer::Request> request) {
        try {
          auto content = json::parse(request->content.string());
          Feedback feedback = content;
          decisionService_->computeRewardAndLogFeedback(feedback);
          json responseJson = {{"status", "OK"}};

          response->write(SimpleWeb::StatusCode::success_ok,
                          responseJson.dump(2));
        } catch (const std::exception& e) {
          LOG(ERROR) << "GOT ERROR: " << e.what();
          response->write(SimpleWeb::StatusCode::client_error_bad_request,
                          e.what());
        }
      };

  server_.on_error = [](std::shared_ptr<HttpServer::Request> request,
                       const SimpleWeb::error_code& ec) {
    // Handle errors here
    // Note that connection timeouts will also call this handle with ec set to
    // SimpleWeb::errc::operation_canceled
    LOG(INFO) << "SERVER ERROR: " << ec.message();
  };

  serverThread_.reset(new std::thread([this]() {
    LOG(INFO) << "STARTING SERVER";
    server_.start();
  }));
}

void Server::shutdown() {
  server_.stop();
  if (serverThread_->joinable()) {
    serverThread_->join();
  }
}

}  // namespace reagent

// From: https://rderik.com/blog/multithreading-with-pthreads-in-swift/
// https://rderik.com/blog/using-bsd-sockets-in-swift/
// https://developer.apple.com/documentation/swift/unsafemutablerawpointer
import Foundation

func threadedFunction(pointer: UnsafeMutableRawPointer) -> UnsafeMutableRawPointer? {
  var threadParameter = pointer.load(as: ThreadParameter.self)
  var server = threadParameter.server
  var threadIdentifier = threadParameter.threadIdentifier
  while let client = server.clients.dequeue() {
    server.processClient(client: client, thread: threadIdentifier.id)
  }
  return nil
}


print("Welcome to our Server simulator")

let server = Server(numberOfClients: 10)
let startTime = CFAbsoluteTimeGetCurrent()
server.start(numberOfThreads: 5)
let endTime = CFAbsoluteTimeGetCurrent()
print("Duration:\(endTime - startTime) seconds")


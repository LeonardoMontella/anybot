// anybot-reconstruct: photos of a place -> textured 3D mesh (OBJ), with Apple Object Capture
// (RealityKit PhotogrammetrySession, runs on the Mac's GPU). Built by anybot.capture on first use.
//
// usage: anybot-reconstruct <images folder> <output folder> [preview|reduced|medium|full]
// prints "progress <0..1>", "skip <sample> <reason>", "done <path>" or "error <message>".
import Foundation
import RealityKit

setvbuf(stdout, nil, _IOLBF, 0)
let args = CommandLine.arguments
guard args.count >= 3 else {
    print("error usage: anybot-reconstruct <images folder> <output folder> [detail]")
    exit(2)
}
guard PhotogrammetrySession.isSupported else {
    print("error Object Capture is not supported on this Mac")
    exit(3)
}
let input = URL(fileURLWithPath: args[1], isDirectory: true)
let output = URL(fileURLWithPath: args[2], isDirectory: true)
let detail: PhotogrammetrySession.Request.Detail
switch args.count > 3 ? args[3] : "reduced" {
case "preview": detail = .preview
case "medium": detail = .medium
case "full": detail = .full
default: detail = .reduced
}
try? FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)

var config = PhotogrammetrySession.Configuration()
config.sampleOrdering = .sequential  // frames of a video: neighbours overlap
config.featureSensitivity = .high  // rooms have plain walls: look harder for features
config.isObjectMaskingEnabled = false  // a whole place, not one object on a turntable

let session: PhotogrammetrySession
do {
    session = try PhotogrammetrySession(input: input, configuration: config)
} catch {
    print("error \(error.localizedDescription)")
    exit(4)
}

Task {
    do {
        for try await out in session.outputs {
            switch out {
            case .requestProgress(_, let fraction):
                print(String(format: "progress %.3f", fraction))
            case .requestComplete(_, let result):
                if case .modelFile(let url) = result { print("done \(url.path)") }
            case .requestError(_, let err):
                print("error \(err.localizedDescription)")
                exit(5)
            case .invalidSample(let id, let reason):
                print("skip \(id) \(reason)")
            case .processingComplete:
                exit(0)
            default:
                break
            }
        }
    } catch {
        print("error \(error.localizedDescription)")
        exit(6)
    }
}
do {
    // A folder URL: the model is written as OBJ + MTL + texture images.
    try session.process(requests: [.modelFile(url: output, detail: detail)])
} catch {
    print("error \(error.localizedDescription)")
    exit(7)
}
RunLoop.main.run()

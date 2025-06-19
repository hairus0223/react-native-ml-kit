#import "FaceDetection.h"
@import MLKitVision;
@import MLKitFaceDetection;

@implementation FaceDetection

RCT_EXPORT_MODULE()

- (MLKFaceDetectorOptions*)getOptions:(NSDictionary*)dict {
    MLKFaceDetectorOptions* options = [[MLKFaceDetectorOptions alloc] init];
    if ([dict[@"performanceMode"] isEqual:@"accurate"]) {
        options.performanceMode = MLKFaceDetectorPerformanceModeAccurate;
    }
    if ([dict[@"landmarkMode"] isEqual:@"all"]) {
        options.landmarkMode = MLKFaceDetectorLandmarkModeAll;
    }
    if ([dict[@"contourMode"] isEqual:@"all"]) {
        options.contourMode = MLKFaceDetectorContourModeAll;
    }
    if ([dict[@"classificationMode"] isEqual:@"all"]) {
        options.classificationMode = MLKFaceDetectorClassificationModeAll;
    }
    if ([dict[@"minFaceSize"] doubleValue] > 0 && [dict[@"minFaceSize"] doubleValue] <= 1) {
        options.minFaceSize = [dict[@"minFaceSize"] doubleValue];
    }
    options.trackingEnabled = [dict[@"trackingEnabled"] boolValue];
    return options;
}

- (NSDictionary*)frameToDict:(CGRect)frame {
    return @{
        @"width": @(frame.size.width),
        @"height": @(frame.size.height),
        @"left": @(frame.origin.x),
        @"top": @(frame.origin.y)
    };
}

- (NSDictionary*)pointToDict:(MLKVisionPoint*)point {
    return @{
        @"x": @(point.x),
        @"y": @(point.y)
    };
}

- (NSDictionary*)landmarkToDict:(MLKFaceLandmark*)landmark {
    return @{
        @"position": [self pointToDict:landmark.position]
    };
}

- (NSDictionary*)contourToDict:(MLKFaceContour*)contour {
    NSMutableArray *points = [NSMutableArray array];
    for (MLKVisionPoint *point in contour.points) {
        [points addObject:[self pointToDict:point]];
    }
    return @{ @"points": points };
}

- (NSDictionary*)faceToDict:(MLKFace*)face withOptions:(MLKFaceDetectorOptions*)options {
    NSMutableDictionary* dict = [NSMutableDictionary dictionary];

    [dict setObject:[self frameToDict:face.frame] forKey:@"frame"];
    
    if (face.hasHeadEulerAngleX) {
        dict[@"rotationX"] = @(face.headEulerAngleX);
    }
    if (face.hasHeadEulerAngleY) {
        dict[@"rotationY"] = @(face.headEulerAngleY);
    }
    if (face.hasHeadEulerAngleZ) {
        dict[@"rotationZ"] = @(face.headEulerAngleZ);
    }

    if (options.landmarkMode == MLKFaceDetectorLandmarkModeAll) {
        NSMutableDictionary *landmarks = [NSMutableDictionary dictionary];
        NSArray *landmarkTypes = @[
            @(MLKFaceLandmarkTypeLeftEar), @(MLKFaceLandmarkTypeRightEar),
            @(MLKFaceLandmarkTypeLeftEye), @(MLKFaceLandmarkTypeRightEye),
            @(MLKFaceLandmarkTypeNoseBase),
            @(MLKFaceLandmarkTypeLeftCheek), @(MLKFaceLandmarkTypeRightCheek),
            @(MLKFaceLandmarkTypeMouthLeft), @(MLKFaceLandmarkTypeMouthRight),
            @(MLKFaceLandmarkTypeMouthBottom)
        ];
        NSArray *landmarkKeys = @[
            @"leftEar", @"rightEar",
            @"leftEye", @"rightEye",
            @"noseBase",
            @"leftCheek", @"rightCheek",
            @"mouthLeft", @"mouthRight",
            @"mouthBottom"
        ];

        for (int i = 0; i < landmarkTypes.count; i++) {
            MLKFaceLandmark *landmark = [face landmarkOfType:[landmarkTypes[i] intValue]];
            if (landmark != nil) {
                landmarks[landmarkKeys[i]] = [self landmarkToDict:landmark];
            }
        }

        dict[@"landmarks"] = landmarks;
    }

    if (options.contourMode == MLKFaceDetectorContourModeAll) {
        NSMutableDictionary *contours = [NSMutableDictionary dictionary];
        NSArray *contourTypes = @[
            @(MLKFaceContourTypeFace), @(MLKFaceContourTypeLeftEye), @(MLKFaceContourTypeRightEye),
            @(MLKFaceContourTypeLeftCheek), @(MLKFaceContourTypeRightCheek),
            @(MLKFaceContourTypeNoseBottom), @(MLKFaceContourTypeNoseBridge),
            @(MLKFaceContourTypeLeftEyebrowTop), @(MLKFaceContourTypeLeftEyebrowBottom),
            @(MLKFaceContourTypeRightEyebrowTop), @(MLKFaceContourTypeRightEyebrowBottom),
            @(MLKFaceContourTypeUpperLipTop), @(MLKFaceContourTypeUpperLipBottom),
            @(MLKFaceContourTypeLowerLipTop), @(MLKFaceContourTypeLowerLipBottom)
        ];
        NSArray *contourKeys = @[
            @"face", @"leftEye", @"rightEye",
            @"leftCheek", @"rightCheek",
            @"noseBottom", @"noseBridge",
            @"leftEyebrowTop", @"leftEyebrowBottom",
            @"rightEyebrowTop", @"rightEyebrowBottom",
            @"upperLipTop", @"upperLipBottom",
            @"lowerLipTop", @"lowerLipBottom"
        ];

        for (int i = 0; i < contourTypes.count; i++) {
            MLKFaceContour *contour = [face contourOfType:[contourTypes[i] intValue]];
            if (contour != nil) {
                contours[contourKeys[i]] = [self contourToDict:contour];
            }
        }

        dict[@"contours"] = contours;
    }

    if (face.hasSmilingProbability) {
        dict[@"smilingProbability"] = @(face.smilingProbability);
    }
    if (face.hasLeftEyeOpenProbability) {
        dict[@"leftEyeOpenProbability"] = @(face.leftEyeOpenProbability);
    }
    if (face.hasRightEyeOpenProbability) {
        dict[@"rightEyeOpenProbability"] = @(face.rightEyeOpenProbability);
    }
    if (face.hasTrackingID) {
        dict[@"trackingID"] = @(face.trackingID);
    }

    return dict;
}

RCT_EXPORT_METHOD(detect:(NSString*)url
                  withOptions:(NSDictionary*)optionsDict
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
    NSURL *_url = [NSURL URLWithString:url];
    NSData *imageData = [NSData dataWithContentsOfURL:_url];
    UIImage *image = [UIImage imageWithData:imageData];

    if (!image) {
        reject(@"no_image", @"Unable to load image", nil);
        return;
    }

    MLKVisionImage *visionImage = [[MLKVisionImage alloc] initWithImage:image];
    visionImage.orientation = image.imageOrientation;

    MLKFaceDetectorOptions *options = [self getOptions:optionsDict];
    MLKFaceDetector *faceDetector = [MLKFaceDetector faceDetectorWithOptions:options];

    [faceDetector processImage:visionImage
                    completion:^(NSArray<MLKFace *> *faces, NSError *error) {
        if (error != nil) {
            reject(@"Face Detection", @"Face detection failed", error);
            return;
        }

        if (faces == nil) {
            resolve(@[]);
            return;
        }

        NSMutableArray *result = [NSMutableArray array];
        for (MLKFace *face in faces) {
            [result addObject:[self faceToDict:face withOptions:options]];
        }
        resolve(result);
    }];
}

@end

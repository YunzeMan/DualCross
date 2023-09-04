from fire import Fire
import src


if __name__ == '__main__':
    Fire({
        'lidar_check': src.explore.lidar_check,
        'cumsum_check': src.explore.cumsum_check,

        'train': src.train.train,
        'test_and_vis': src.test_and_vis.test_and_vis,
        'eval_model_iou': src.explore.eval_model_iou,
        'viz_model_preds': src.explore.viz_model_preds,
        'viz_bad_samples': src.explore.viz_bad_samples,
        'viz_vehicle_road_lane': src.visualize.viz_vehicle_road_lane,
        'make_video': src.visualize.make_video,
    })
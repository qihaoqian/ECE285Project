from tensorboard.backend.event_processing import event_accumulator

# 替换为你的实际路径
path = "workspace/ngp/bed_0248/run/ngp/events.out.tfevents.1749282158.5070ti"

ea = event_accumulator.EventAccumulator(path)
ea.Reload()

# 查看包含哪些 scalar tag
print(ea.Tags()["scalars"])

# 假设你想看训练损失 'train/loss'
events = ea.Scalars("train/loss")

start_time = events[0].wall_time
end_time = events[-1].wall_time
duration_seconds = end_time - start_time

import datetime
print("训练持续时间：", str(datetime.timedelta(seconds=duration_seconds)))

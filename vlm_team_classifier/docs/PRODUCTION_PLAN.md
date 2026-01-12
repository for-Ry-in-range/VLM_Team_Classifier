# Production Plan

## Deployment Strategy

My deployment strategy is to use AWS ECS with Fargate. We will use an API gateway for users to upload videos. The video will be stored in S3. Then a container will see the new video, load YOLO and SigLIP, and start processing. We will use Redis to remember previously seen uniforms in order to decrease redudant inferences. The extracted data will be saved in DynamoDB and the annotated video will be saved in S3. 

I chose to use ECS instead of Lambda because ECS can keep large models like YOLO and SigLIP in memory, so you don't have to wait 20 seconds every time before processsing a video. In addition, Lambda has a 15 minute time limit before it shuts down, which would not work when processing long basketball games. Furthermore, ECS offers the CPU and GPU power needed for AI and computer vision that Lambda does not have.

I will use auto-scaling based on the queue depth. For example, if 8 videos are uploaded at the same time, the system can start up more containers to process the videos faster.


## Monitoring & Observability

**Metrics to track:**
- **Performance:** Latency, FPS, CPU/memory usage
- **Accuracy:** Precision, recall, accuracy
- **Reliability:** Error rate, uptime, failed job rate
- **Cost:** Cost per frame, cost per day

We can track these metrics using CloudWatch for collecting the raw data, Datadog for displaying dashboards, and PagerDuty for alerts. Alerts can be turned on for accuracy under 90% and a queue depth above 100.

## Failure Handling

If YOLO is not detecting enough people, we can lower the confidence threshold. If the hybrid of SigLIP and RGB fails, we can fallback to the color-only baseline. If the baseline manages to fail, we can flag the video for manual review.

If the container crashes, we will auto-restart so that no manual work is needed. If the system is running out of memory, we can process videos in short segments to prevent too many frames from being stored in memory at once. Then, in between segments the memory will be released. 


## Cost Analysis

If each game footage is 48 minutes at 30 FPS, then we have 86,400 frames per game.

**\*My computer is using the Intel Core i5-8210Y, which is estimated to run models about 750 times slower than the NVIDIA H100 (which is what we are using)**

**Estimated cost per day at 1000 games/day ($2/hr):** $3,200


## Iteration Plan

**Phase 1:**
- Deploy to production and achieve at least 90% accuracy
- Process at least 1,000 minutes of video per day
- Basic monitoring and error handling

**Phase 2:**
- Reduce latency and cost by 25%
- Scale to 5,000 minutes of video processed per day
- Implement model quantization to increase speed

**Phase 3:**
- Track the ball too
- Analyze players' actions and track their statistics
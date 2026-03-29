def fusion_decision(image_score, video_score, audio_score):
    final_score = (
        0.2 * image_score +
        0.5 * video_score +
        0.3 * audio_score
    )

    label = "FAKE" if final_score > 0.5 else "REAL"

    return {
        "final_score": round(final_score, 3),
        "decision": label
    }
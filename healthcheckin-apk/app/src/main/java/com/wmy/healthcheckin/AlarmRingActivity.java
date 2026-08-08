package com.wmy.healthcheckin;

import android.app.Activity;
import android.app.NotificationManager;
import android.graphics.Color;
import android.media.Ringtone;
import android.media.RingtoneManager;
import android.os.Build;
import android.os.Bundle;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.view.Gravity;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

public class AlarmRingActivity extends Activity {
    private Ringtone ringtone;
    private Vibrator vibrator;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON |
                WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD |
                WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED |
                WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON);
        if (Build.VERSION.SDK_INT >= 27) {
            setShowWhenLocked(true);
            setTurnScreenOn(true);
        }

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setGravity(Gravity.CENTER);
        root.setPadding(44, 60, 44, 60);
        root.setBackgroundColor(Color.rgb(239, 248, 242));

        TextView icon = new TextView(this);
        icon.setText("☀"); icon.setTextSize(64); icon.setGravity(Gravity.CENTER);
        root.addView(icon);

        TextView title = new TextView(this);
        title.setText(getIntent().getStringExtra("title") == null ? "起床时间到" : getIntent().getStringExtra("title"));
        title.setTextSize(32); title.setTextColor(Color.rgb(23,32,25)); title.setGravity(Gravity.CENTER);
        title.setPadding(0, 24, 0, 12);
        root.addView(title);

        TextView msg = new TextView(this);
        msg.setText(getIntent().getStringExtra("message") == null ? "新的一天开始了" : getIntent().getStringExtra("message"));
        msg.setTextSize(17); msg.setTextColor(Color.rgb(82,97,88)); msg.setGravity(Gravity.CENTER);
        msg.setPadding(0, 0, 0, 42);
        root.addView(msg);

        Button stop = new Button(this);
        stop.setText("我起床了 · 关闭闹钟");
        stop.setTextSize(18);
        stop.setOnClickListener(v -> finish());
        root.addView(stop, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        setContentView(root);

        try {
            ringtone = RingtoneManager.getRingtone(this, RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM));
            if (Build.VERSION.SDK_INT >= 28) ringtone.setLooping(true);
            ringtone.play();
        } catch (Exception ignored) {}
        vibrator = (Vibrator) getSystemService(VIBRATOR_SERVICE);
        if (vibrator != null) {
            long[] pattern = {0, 700, 300, 700, 300};
            if (Build.VERSION.SDK_INT >= 26) vibrator.vibrate(VibrationEffect.createWaveform(pattern, 0));
            else vibrator.vibrate(pattern, 0);
        }
    }

    @Override
    protected void onDestroy() {
        if (ringtone != null && ringtone.isPlaying()) ringtone.stop();
        if (vibrator != null) vibrator.cancel();
        NotificationManager nm = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        nm.cancel(6003);
        super.onDestroy();
    }
}

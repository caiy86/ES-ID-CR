package com.wmy.healthcheckin;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.media.AudioAttributes;
import android.media.RingtoneManager;
import android.net.Uri;
import android.os.Build;

public class AlarmReceiver extends BroadcastReceiver {
    @Override
    public void onReceive(Context context, Intent intent) {
        String type = intent.getStringExtra("type");
        String title = intent.getStringExtra("title");
        String message = intent.getStringExtra("message");
        if (type == null) type = "reminder";
        if (title == null) title = "身体恢复提醒";
        if (message == null) message = "该休息一下了。";

        NotificationManager nm = (NotificationManager) context.getSystemService(Context.NOTIFICATION_SERVICE);
        boolean wake = "wake".equals(type);
        String channelId = wake ? "wake_alarm_v2" : "sleep_reminder_v2";

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(channelId,
                    wake ? "起床闹钟" : "睡眠提醒",
                    wake ? NotificationManager.IMPORTANCE_HIGH : NotificationManager.IMPORTANCE_DEFAULT);
            channel.enableVibration(true);
            channel.setVibrationPattern(wake ? new long[]{0, 500, 250, 500, 250, 700} : new long[]{0, 180, 120, 180});
            if (wake) {
                Uri sound = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM);
                AudioAttributes attrs = new AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_ALARM).build();
                channel.setSound(sound, attrs);
            }
            nm.createNotificationChannel(channel);
        }

        Intent openIntent = wake ? new Intent(context, AlarmRingActivity.class) : new Intent(context, MainActivity.class);
        openIntent.putExtra("title", title);
        openIntent.putExtra("message", message);
        PendingIntent openPi = PendingIntent.getActivity(context, wake ? 5201 : 5202, openIntent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);

        Notification.Builder b = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                ? new Notification.Builder(context, channelId)
                : new Notification.Builder(context);
        b.setSmallIcon(com.wmy.healthcheckin.R.drawable.ic_launcher)
                .setContentTitle(title)
                .setContentText(message)
                .setContentIntent(openPi)
                .setAutoCancel(!wake)
                .setCategory(wake ? Notification.CATEGORY_ALARM : Notification.CATEGORY_REMINDER)
                .setVisibility(Notification.VISIBILITY_PUBLIC)
                .setColor(Color.rgb(47, 111, 78));
        if (wake) {
            b.setOngoing(true).setFullScreenIntent(openPi, true).setPriority(Notification.PRIORITY_MAX);
        } else {
            b.setPriority(Notification.PRIORITY_HIGH);
        }
        nm.notify(wake ? 6003 : ("hour".equals(type) ? 6001 : 6002), b.build());
        AlarmScheduler.rescheduleType(context, type);
    }
}

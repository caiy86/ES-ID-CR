package com.wmy.healthcheckin;

import android.app.AlarmManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Build;

import java.util.Calendar;

public final class AlarmScheduler {
    private static final String PREF = "alarm_settings";
    private static final int REQ_HOUR = 4101;
    private static final int REQ_HALF = 4102;
    private static final int REQ_WAKE = 4103;

    private AlarmScheduler() {}

    public static void saveAndSchedule(Context context, String bedtime, String wake,
                                       boolean hourEnabled, boolean halfEnabled, boolean wakeEnabled) {
        context.getSharedPreferences(PREF, Context.MODE_PRIVATE).edit()
                .putString("bedtime", bedtime)
                .putString("wake", wake)
                .putBoolean("hour", hourEnabled)
                .putBoolean("half", halfEnabled)
                .putBoolean("wakeEnabled", wakeEnabled)
                .apply();
        scheduleFromPrefs(context);
    }

    public static String getSettingsJson(Context context) {
        SharedPreferences p = context.getSharedPreferences(PREF, Context.MODE_PRIVATE);
        String bedtime = p.getString("bedtime", "23:00");
        String wake = p.getString("wake", "06:30");
        return "{\"bedtime\":\"" + bedtime + "\",\"wake\":\"" + wake +
                "\",\"hour\":" + p.getBoolean("hour", true) +
                ",\"half\":" + p.getBoolean("half", true) +
                ",\"wakeEnabled\":" + p.getBoolean("wakeEnabled", true) + "}";
    }

    public static void scheduleFromPrefs(Context context) {
        SharedPreferences p = context.getSharedPreferences(PREF, Context.MODE_PRIVATE);
        cancelAll(context);
        String bedtime = p.getString("bedtime", "23:00");
        String wake = p.getString("wake", "06:30");
        if (p.getBoolean("hour", true)) {
            schedule(context, REQ_HOUR, nextTime(bedtime, -60), "hour", "准备收尾", "距离睡觉还有1小时，开始减少刺激和工作强度。", false);
        }
        if (p.getBoolean("half", true)) {
            schedule(context, REQ_HALF, nextTime(bedtime, -30), "half", "准备睡觉", "距离睡觉还有30分钟，洗漱、放下手机、让大脑慢下来。", false);
        }
        if (p.getBoolean("wakeEnabled", true)) {
            schedule(context, REQ_WAKE, nextTime(wake, 0), "wake", "起床时间到", "新的一天开始了，起床喝水、活动一下。", true);
        }
    }

    public static void rescheduleType(Context context, String type) {
        SharedPreferences p = context.getSharedPreferences(PREF, Context.MODE_PRIVATE);
        if ("hour".equals(type) && p.getBoolean("hour", true)) {
            schedule(context, REQ_HOUR, nextTime(p.getString("bedtime", "23:00"), -60), "hour", "准备收尾", "距离睡觉还有1小时，开始减少刺激和工作强度。", false);
        } else if ("half".equals(type) && p.getBoolean("half", true)) {
            schedule(context, REQ_HALF, nextTime(p.getString("bedtime", "23:00"), -30), "half", "准备睡觉", "距离睡觉还有30分钟，洗漱、放下手机、让大脑慢下来。", false);
        } else if ("wake".equals(type) && p.getBoolean("wakeEnabled", true)) {
            schedule(context, REQ_WAKE, nextTime(p.getString("wake", "06:30"), 0), "wake", "起床时间到", "新的一天开始了，起床喝水、活动一下。", true);
        }
    }

    private static long nextTime(String hhmm, int offsetMinutes) {
        String[] parts = hhmm.split(":");
        int hour = Integer.parseInt(parts[0]);
        int minute = Integer.parseInt(parts[1]);
        Calendar c = Calendar.getInstance();
        c.set(Calendar.HOUR_OF_DAY, hour);
        c.set(Calendar.MINUTE, minute);
        c.set(Calendar.SECOND, 0);
        c.set(Calendar.MILLISECOND, 0);
        c.add(Calendar.MINUTE, offsetMinutes);
        if (c.getTimeInMillis() <= System.currentTimeMillis() + 1000) c.add(Calendar.DAY_OF_YEAR, 1);
        return c.getTimeInMillis();
    }

    private static void schedule(Context context, int requestCode, long when, String type,
                                 String title, String message, boolean alarmClock) {
        AlarmManager am = (AlarmManager) context.getSystemService(Context.ALARM_SERVICE);
        Intent intent = new Intent(context, AlarmReceiver.class)
                .putExtra("type", type)
                .putExtra("title", title)
                .putExtra("message", message);
        PendingIntent operation = PendingIntent.getBroadcast(context, requestCode, intent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);

        if (alarmClock) {
            PendingIntent show = PendingIntent.getActivity(context, 4999,
                    new Intent(context, MainActivity.class),
                    PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
            am.setAlarmClock(new AlarmManager.AlarmClockInfo(when, show), operation);
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            am.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, when, operation);
        } else {
            am.setExact(AlarmManager.RTC_WAKEUP, when, operation);
        }
    }

    private static void cancelAll(Context context) {
        AlarmManager am = (AlarmManager) context.getSystemService(Context.ALARM_SERVICE);
        for (int code : new int[]{REQ_HOUR, REQ_HALF, REQ_WAKE}) {
            PendingIntent pi = PendingIntent.getBroadcast(context, code,
                    new Intent(context, AlarmReceiver.class),
                    PendingIntent.FLAG_NO_CREATE | PendingIntent.FLAG_IMMUTABLE);
            if (pi != null) {
                am.cancel(pi);
                pi.cancel();
            }
        }
    }
}

package com.wmy.healthcheckin;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.webkit.JavascriptInterface;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Toast;

import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

public class MainActivity extends Activity {
    private static final int FILE_CHOOSER_REQUEST = 2001;
    private static final int EXPORT_REQUEST = 2002;
    private WebView webView;
    private ValueCallback<Uri[]> filePathCallback;
    private String pendingExportJson;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setStatusBarColor(Color.rgb(246, 248, 246));
        getWindow().getDecorView().setSystemUiVisibility(android.view.View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR);

        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 3301);
        }
        AlarmScheduler.scheduleFromPrefs(this);

        webView = new WebView(this);
        WebSettings s = webView.getSettings();
        s.setJavaScriptEnabled(true); s.setDomStorageEnabled(true); s.setAllowFileAccess(true); s.setAllowContentAccess(true);
        webView.setWebViewClient(new WebViewClient());
        webView.setWebChromeClient(new WebChromeClient() {
            @Override public boolean onShowFileChooser(WebView webView, ValueCallback<Uri[]> cb, FileChooserParams params) {
                if (filePathCallback != null) filePathCallback.onReceiveValue(null);
                filePathCallback = cb;
                Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("application/json");
                startActivityForResult(i, FILE_CHOOSER_REQUEST); return true;
            }
        });
        webView.addJavascriptInterface(new Bridge(), "Android");
        setContentView(webView);
        webView.loadUrl("file:///android_asset/index.html");
    }

    private class Bridge {
        @JavascriptInterface public void saveBackup(String json) {
            runOnUiThread(() -> {
                pendingExportJson = json;
                Intent i = new Intent(Intent.ACTION_CREATE_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("application/json");
                i.putExtra(Intent.EXTRA_TITLE, "身体恢复打卡_备份.json"); startActivityForResult(i, EXPORT_REQUEST);
            });
        }
        @JavascriptInterface public void scheduleAlarms(String bedtime, String wake, boolean hour, boolean half, boolean wakeEnabled) {
            AlarmScheduler.saveAndSchedule(MainActivity.this, bedtime, wake, hour, half, wakeEnabled);
            runOnUiThread(() -> Toast.makeText(MainActivity.this, "闹钟设置已生效", Toast.LENGTH_SHORT).show());
        }
        @JavascriptInterface public String getAlarmSettings() { return AlarmScheduler.getSettingsJson(MainActivity.this); }
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == FILE_CHOOSER_REQUEST) {
            if (filePathCallback == null) return; Uri[] result = null;
            if (resultCode == RESULT_OK && data != null && data.getData() != null) result = new Uri[]{data.getData()};
            filePathCallback.onReceiveValue(result); filePathCallback = null; return;
        }
        if (requestCode == EXPORT_REQUEST && resultCode == RESULT_OK && data != null && data.getData() != null) {
            try (OutputStream out = getContentResolver().openOutputStream(data.getData())) {
                if (out != null && pendingExportJson != null) { out.write(pendingExportJson.getBytes(StandardCharsets.UTF_8)); out.flush(); Toast.makeText(this,"备份已保存",Toast.LENGTH_SHORT).show(); }
            } catch (Exception e) { Toast.makeText(this,"备份保存失败",Toast.LENGTH_SHORT).show(); }
            pendingExportJson = null;
        }
    }

    @Override public void onBackPressed() { if (webView != null && webView.canGoBack()) webView.goBack(); else super.onBackPressed(); }
    @Override protected void onDestroy() { if (webView != null) webView.destroy(); super.onDestroy(); }
}

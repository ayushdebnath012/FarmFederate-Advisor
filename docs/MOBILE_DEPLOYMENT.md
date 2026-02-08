# FarmFederate Mobile Deployment Guide

This guide explains how to deploy the FarmFederate backend and build the mobile app.

## Quick Start: Deploy Backend to Render (Free)

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

### Step 2: Deploy to Render
1. Go to [render.com](https://render.com) and sign up (free, no credit card)
2. Click **New** > **Web Service**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `farmfederate-api`
   - **Region**: Oregon (or closest to you)
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements-deploy.txt`
   - **Start Command**: `cd backend && uvicorn server:app --host 0.0.0.0 --port $PORT`
5. Add Environment Variable:
   - `DEMO_MODE` = `true` (for free tier without ML models)
6. Click **Create Web Service**

Your backend will be live at: `https://farmfederate-api.onrender.com`

### Step 3: Update Flutter App
Edit `frontend/lib/constants.dart`:
```dart
const String PRODUCTION_BACKEND = "https://farmfederate-api.onrender.com";
const bool IS_PRODUCTION = true;
```

---

## Prerequisites

- Flutter SDK 3.0+
- For Android: Android Studio with Android SDK
- For iOS: Xcode (macOS only)

## Configuration

### Backend URL Configuration

Edit `frontend/lib/constants.dart`:

```dart
// For production deployment, update these values:
const String PRODUCTION_BACKEND = "https://your-api-server.com";
const bool IS_PRODUCTION = true;
```

**Development options:**
- Android Emulator: Uses `http://10.0.2.2:8000` automatically
- iOS Simulator: Uses `http://localhost:8000` automatically
- Real device testing: Update `_LAN_BACKEND` to your computer's local IP

### 2. Firebase Configuration (Optional)

If using Firebase authentication:
1. Create a Firebase project at https://console.firebase.google.com
2. Download `google-services.json` (Android) to `frontend/android/app/`
3. Download `GoogleService-Info.plist` (iOS) to `frontend/ios/Runner/`

## Building for Android

### Debug APK (for testing)
```bash
cd frontend
flutter build apk --debug
```
Output: `build/app/outputs/flutter-apk/app-debug.apk`

### Release APK
```bash
cd frontend
flutter build apk --release
```
Output: `build/app/outputs/flutter-apk/app-release.apk`

### App Bundle (for Play Store)
```bash
cd frontend
flutter build appbundle --release
```
Output: `build/app/outputs/bundle/release/app-release.aab`

### Install on connected device
```bash
flutter install
```

## Building for iOS

**Note:** iOS builds require macOS with Xcode installed.

### Debug build
```bash
cd frontend
flutter build ios --debug
```

### Release build
```bash
cd frontend
flutter build ios --release
```

### Open in Xcode (for device deployment/App Store)
```bash
open ios/Runner.xcworkspace
```

## Testing on Real Devices

### Android
1. Enable Developer Options on your device
2. Enable USB Debugging
3. Connect device via USB
4. Run: `flutter devices` to verify connection
5. Run: `flutter run` or `flutter install`

### iOS
1. Connect device to Mac
2. Open `ios/Runner.xcworkspace` in Xcode
3. Select your device as target
4. Click Run or Archive for distribution

## App Signing

### Android Release Signing
1. Generate a keystore:
   ```bash
   keytool -genkey -v -keystore ~/farmfederate-release.jks -keyalg RSA -keysize 2048 -validity 10000 -alias farmfederate
   ```

2. Create `frontend/android/key.properties`:
   ```properties
   storePassword=<password>
   keyPassword=<password>
   keyAlias=farmfederate
   storeFile=<path-to-keystore>
   ```

3. Update `frontend/android/app/build.gradle.kts` to use the keystore for release builds.

### iOS Signing
Configure in Xcode under Signing & Capabilities with your Apple Developer account.

## Store Deployment

### Google Play Store
1. Build app bundle: `flutter build appbundle`
2. Go to [Google Play Console](https://play.google.com/console)
3. Create new app and upload `.aab` file
4. Fill in store listing details
5. Submit for review

### Apple App Store
1. Build in Xcode: Product > Archive
2. Upload to App Store Connect via Xcode or Transporter
3. Fill in app metadata
4. Submit for review

## Troubleshooting

### "Network error" on real device
- Ensure backend is accessible from device network
- Update `_LAN_BACKEND` in constants.dart with your computer's IP
- Check firewall allows connections on port 8000

### Camera not working
- Android: Permissions are in AndroidManifest.xml (already configured)
- iOS: Usage descriptions are in Info.plist (already configured)
- Ensure user grants permission when prompted

### Firebase errors
- Verify google-services.json (Android) is in correct location
- Verify GoogleService-Info.plist (iOS) is in correct location
- Check Firebase project configuration matches app bundle ID

## App Configuration

| Setting | Value |
|---------|-------|
| Android Package | `com.farmfederate.app` |
| iOS Bundle ID | Update in Xcode |
| Min Android SDK | 21 (Android 5.0) |
| Min iOS Version | 12.0 |

## Quick Commands

```bash
# Get dependencies
flutter pub get

# Run on connected device
flutter run

# Build debug APK
flutter build apk --debug

# Build release APK
flutter build apk --release

# List connected devices
flutter devices

# Clean build artifacts
flutter clean
```

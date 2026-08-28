# Firebase Setup Guide for FarmFederate-Advisor

## Prerequisites
1. A Google account
2. Flutter SDK installed
3. Firebase CLI installed: `npm install -g firebase-tools`

## Step 1: Create Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Add project"
3. Enter project name: `farmfederate-advisor`
4. Follow the setup wizard
5. Enable Google Analytics (optional)

## Step 2: Enable Authentication

1. In Firebase Console, go to **Build** -> **Authentication**
2. Click "Get started"
3. Click "Sign-in method" tab
4. Enable **Email/Password** provider
5. Click "Save"

## Step 3: Add Web App to Firebase

1. In Firebase Console, click the **Web** icon (`</>`) to add a web app
2. Register app with nickname: `FarmFederate Web`
3. Copy the Firebase configuration object
4. Replace values in `frontend/lib/main.dart`:

```dart
await Firebase.initializeApp(
  options: const FirebaseOptions(
    apiKey: "YOUR_API_KEY",
    appId: "YOUR_APP_ID",
    messagingSenderId: "YOUR_SENDER_ID",
    projectId: "YOUR_PROJECT_ID",
    authDomain: "YOUR_PROJECT_ID.firebaseapp.com",
    storageBucket: "YOUR_PROJECT_ID.appspot.com",
  ),
);
```

## Step 4: Install Dependencies

```powershell
cd frontend
flutter pub get
```

## Step 5: Configure for Android (Optional)

1. In Firebase Console, click the Android icon
2. Register app with package name from `android/app/build.gradle`
3. Download `google-services.json`
4. Place it in `frontend/android/app/`
5. Update `android/build.gradle`:

```gradle
buildscript {
  dependencies {
    classpath 'com.google.gms:google-services:4.3.15'
  }
}
```

6. Update `android/app/build.gradle`:

```gradle
apply plugin: 'com.google.gms.google-services'
```

## Step 6: Configure for iOS (Optional)

1. In Firebase Console, click the iOS icon
2. Register app with bundle ID from `ios/Runner.xcodeproj`
3. Download `GoogleService-Info.plist`
4. Drag it into `ios/Runner` in Xcode

## Step 7: Run the App

```powershell
flutter run -d chrome

flutter run -d <android-device-id>

flutter run -d <ios-device-id>
```

## Troubleshooting

### "Firebase not initialized" error
- Ensure Firebase.initializeApp() is called before runApp()
- Check that FirebaseOptions values are correct

### Authentication errors
- Verify Email/Password provider is enabled in Firebase Console
- Check Firebase rules allow read/write access

### Build errors
- Run `flutter clean`
- Run `flutter pub get`
- Delete `build/` folder and rebuild

## Security Notes

1. **Never commit Firebase credentials to Git**
   - Add to `.gitignore`: `google-services.json`, `GoogleService-Info.plist`
   - Use environment variables or Firebase Remote Config for production

2. **Set up Firebase Security Rules**
   ```
   rules_version = '2';
   service cloud.firestore {
     match /databases/{database}/documents {
       match /{document=**} {
         allow read, write: if request.auth != null;
       }
     }
   }
   ```

3. **Enable App Check** (production)
   - Protects backend from abuse
   - Verifies requests come from your authentic app

## Testing Users

Create test users in Firebase Console -> Authentication -> Users:
- Click "Add user"
- Enter email and password
- Use these credentials to test login

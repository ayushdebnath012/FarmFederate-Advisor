import 'package:firebase_auth/firebase_auth.dart';
import 'package:shared_preferences/shared_preferences.dart';

class AuthService {
  final FirebaseAuth _auth = FirebaseAuth.instance;

  User? get currentUser => _auth.currentUser;

  Stream<User?> get authStateChanges => _auth.authStateChanges();

  Future<Map<String, dynamic>> signIn(String email, String password) async {
    try {
      final UserCredential result = await _auth.signInWithEmailAndPassword(
        email: email,
        password: password,
      );

      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool('isLoggedIn', true);
      await prefs.setString('userEmail', email);

      return {
        'success': true,
        'user': result.user,
        'message': 'Login successful',
      };
    } on FirebaseAuthException catch (e) {
      String message = 'An error occurred';

      switch (e.code) {
        case 'user-not-found':
          message = 'No user found with this email';
          break;
        case 'wrong-password':
          message = 'Wrong password provided';
          break;
        case 'invalid-email':
          message = 'Invalid email address';
          break;
        case 'user-disabled':
          message = 'This user account has been disabled';
          break;
        case 'too-many-requests':
          message = 'Too many attempts. Please try again later';
          break;
        default:
          message = e.message ?? 'Login failed';
      }

      return {'success': false, 'message': message};
    } catch (e) {
      return {
        'success': false,
        'message': 'An unexpected error occurred: ${e.toString()}',
      };
    }
  }

  Future<Map<String, dynamic>> register(
    String email,
    String password,
    String displayName,
  ) async {
    try {
      final UserCredential result = await _auth.createUserWithEmailAndPassword(
        email: email,
        password: password,
      );

      await result.user?.updateDisplayName(displayName);

      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool('isLoggedIn', true);
      await prefs.setString('userEmail', email);

      return {
        'success': true,
        'user': result.user,
        'message': 'Registration successful',
      };
    } on FirebaseAuthException catch (e) {
      String message = 'An error occurred';

      switch (e.code) {
        case 'weak-password':
          message = 'The password is too weak. Use at least 6 characters';
          break;
        case 'email-already-in-use':
          message = 'An account already exists with this email';
          break;
        case 'invalid-email':
          message = 'Invalid email address';
          break;
        case 'operation-not-allowed':
          message = 'Email/password accounts are not enabled';
          break;
        default:
          message = e.message ?? 'Registration failed';
      }

      return {'success': false, 'message': message};
    } catch (e) {
      return {
        'success': false,
        'message': 'An unexpected error occurred: ${e.toString()}',
      };
    }
  }

  Future<void> signOut() async {
    await _auth.signOut();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('isLoggedIn', false);
    await prefs.remove('userEmail');
  }

  Future<Map<String, dynamic>> resetPassword(String email) async {
    try {
      await _auth.sendPasswordResetEmail(email: email);
      return {
        'success': true,
        'message': 'Password reset email sent. Please check your inbox.',
      };
    } on FirebaseAuthException catch (e) {
      String message = 'An error occurred';

      switch (e.code) {
        case 'user-not-found':
          message = 'No user found with this email';
          break;
        case 'invalid-email':
          message = 'Invalid email address';
          break;
        default:
          message = e.message ?? 'Failed to send reset email';
      }

      return {'success': false, 'message': message};
    } catch (e) {
      return {
        'success': false,
        'message': 'An unexpected error occurred: ${e.toString()}',
      };
    }
  }

  Future<bool> isLoggedIn() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool('isLoggedIn') ?? false;
  }
}

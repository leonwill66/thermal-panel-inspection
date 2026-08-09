from __future__ import annotations


class TestLogin:
    def test_wrong_password_rejected(self, client, make_user):
        make_user("alice", role="admin", password="correct-horse")
        resp = client.post("/api/login", data={"username": "alice", "password": "wrong"})
        assert resp.status_code == 401

    def test_unknown_user_rejected(self, client):
        resp = client.post("/api/login", data={"username": "nobody", "password": "whatever"})
        assert resp.status_code == 401

    def test_correct_password_logs_in(self, client, make_user):
        make_user("alice", role="admin", password="correct-horse")
        resp = client.post("/api/login", data={"username": "alice", "password": "correct-horse"})
        assert resp.status_code == 200
        assert resp.json() == {"username": "alice", "role": "admin"}

    def test_deactivated_account_rejected(self, client, make_user):
        make_user("alice", role="admin", password="correct-horse", active=False)
        resp = client.post("/api/login", data={"username": "alice", "password": "correct-horse"})
        assert resp.status_code == 401

    def test_session_persists_across_requests(self, client, make_user):
        make_user("alice", role="viewer", password="correct-horse")
        client.post("/api/login", data={"username": "alice", "password": "correct-horse"})
        resp = client.get("/api/me")
        assert resp.status_code == 200
        assert resp.json()["username"] == "alice"

    def test_logout_clears_session(self, client, make_user):
        make_user("alice", role="viewer", password="correct-horse")
        client.post("/api/login", data={"username": "alice", "password": "correct-horse"})
        client.post("/api/logout")
        resp = client.get("/api/me")
        assert resp.status_code == 401

    def test_rate_limited_after_too_many_failures(self, client, make_user):
        make_user("alice", role="admin", password="correct-horse")
        for _ in range(5):
            resp = client.post("/api/login", data={"username": "alice", "password": "wrong"})
            assert resp.status_code == 401
        resp = client.post("/api/login", data={"username": "alice", "password": "correct-horse"})
        assert resp.status_code == 429

    def test_successful_login_clears_failure_count(self, client, make_user):
        make_user("alice", role="admin", password="correct-horse")
        for _ in range(4):
            client.post("/api/login", data={"username": "alice", "password": "wrong"})
        ok = client.post("/api/login", data={"username": "alice", "password": "correct-horse"})
        assert ok.status_code == 200
        # 4 prior failures got cleared by the success above, so 4 more
        # shouldn't trip the 5-attempt limit.
        for _ in range(4):
            resp = client.post("/api/login", data={"username": "alice", "password": "wrong"})
            assert resp.status_code == 401


class TestAccessControl:
    def test_me_requires_login(self, client):
        resp = client.get("/api/me")
        assert resp.status_code == 401

    def test_viewer_cannot_analyze(self, client, make_user):
        make_user("bob", role="viewer", password="password123")
        client.post("/api/login", data={"username": "bob", "password": "password123"})
        resp = client.post("/api/analyze", files=[("files", ("a.jpg", b"not a real image", "image/jpeg"))])
        assert resp.status_code == 403

    def test_inspector_can_reach_analyze_role_check(self, client, make_user, mock_radiometric):
        # Role check should pass for inspector (a real request may still 200
        # or fail later for other reasons, but never 403).
        make_user("carol", role="inspector", password="password123")
        client.post("/api/login", data={"username": "carol", "password": "password123"})
        resp = client.post("/api/analyze", files=[("files", ("a.jpg", b"fake bytes", "image/jpeg"))])
        assert resp.status_code != 403
